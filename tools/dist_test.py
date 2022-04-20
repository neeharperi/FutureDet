import numpy as np
import argparse
import copy
import json
import os
import sys


sys.path.append('/home/nperi/Workspace/FutureDet')
sys.path.append('/home/nperi/Workspace/Core/nuscenes-forecast/python-sdk')


try:
    import apex
except:
    print("No APEX!")
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 

from nuscenes.nuscenes import NuScenes

import pdb 

def save_pred(pred, root, split, modelCheckPoint):
    with open(os.path.join(root, "prediction_{}_{}.pkl".format(split, modelCheckPoint)), "wb") as f:
        pickle.dump(pred, f)

def load_pred(root, split, modelCheckPoint):
    with open(os.path.join(root, "prediction_{}_{}.pkl".format(split, modelCheckPoint)), "rb") as f:
        pred = pickle.load(f)
        return pred 

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument("--root", default="/ssd0/nperi/nuScenes/"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--extractBox", action="store_true")
    parser.add_argument("--forecast", type=int, default=6)
    parser.add_argument("--forecast_mode", default="velocity_forward")
    parser.add_argument("--classname", default="car")
    parser.add_argument("--rerank", default="last")

    parser.add_argument("--tp_pct", type=float, default=0.6)
    parser.add_argument("--static_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--cohort_analysis", action="store_true")
    parser.add_argument("--jitter", action="store_true")
    parser.add_argument("--association_oracle", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--nogroup", action="store_true")

    parser.add_argument("--K", default=1, type=int)
    parser.add_argument("--C", default=1, type=float)

    parser.add_argument("--split", default="val")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--modelCheckPoint", default="latest")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args

    
def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        if args.split == "val" or args.split == "mini_val":
            print("Use Val Set")
            dataset = build_dataset(cfg.data.val)
        else:
            print("Use Train Set")
            cfg.data.val.info_path = cfg.data.val.info_path.replace("infos_val_10sweeps_withvelo_filter_True", "infos_train_10sweeps_withvelo_filter_True")
            cfg.data.val.ann_file = cfg.data.val.info_path.replace("infos_val_10sweeps_withvelo_filter_True", "infos_train_10sweeps_withvelo_filter_True")
            dataset = build_dataset(cfg.data.val)

    if args.extractBox:
        nusc = NuScenes(version=args.version, dataroot=args.root, verbose=False)
        sample_data = [s for s in nusc.sample]
        scene_tokens = [s["scene_token"] for s in nusc.sample]

        scene_data = {}

        for sample, scene in zip(sample_data, scene_tokens):
            if scene not in scene_data.keys():
                scene_data[scene] = []

            scene_data[scene].append(sample)

        data_loader = build_dataloader(
            dataset,
            batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )

        checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

        # put model on gpus
        if distributed:
            model = apex.parallel.convert_syncbn_model(model)
            model = DistributedDataParallel(
                model.cuda(cfg.local_rank),
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                # broadcast_buffers=False,
                find_unused_parameters=True,
            )
        else:
            # model = fuse_bn_recursively(model)
            model = model.cuda()

        model.eval()

        logger.info(f"work dir: {args.work_dir}")
        if cfg.local_rank == 0:
            prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

        detections = {}
        cpu_device = torch.device("cpu")

        start = time.time()

        start = int(len(dataset) / 3)
        end = int(len(dataset) * 2 /3)

        time_start = 0 
        time_end = 0 

        for i, data_batch in enumerate(data_loader):
            if i == start:
                torch.cuda.synchronize()
                time_start = time.time()

            if i == end:
                torch.cuda.synchronize()
                time_end = time.time()

            with torch.no_grad():
                outputs = batch_processor(
                    model, data_batch, train_mode=False, local_rank=args.local_rank,
                )
            for output in outputs:
                token = output["metadata"]["token"]
                for k, v in output.items():
                    if k not in [
                        "metadata",
                    ]:
                        output[k] = v.to(cpu_device)
                detections.update(
                    {token: output,}
                )
                if args.local_rank == 0:
                    prog_bar.update()

        synchronize()

        all_predictions = all_gather(detections)
        
        print("\n Total time per frame: ", (time_end -  time_start) / (end - start))

        if args.local_rank != 0:
            return

        predictions = {}
        for p in all_predictions:
            predictions.update(p)

        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)
        
        save_pred(predictions, args.work_dir, args.split, args.modelCheckPoint)

    if args.local_rank != 0:
        return
    
    predictions = load_pred(args.work_dir, args.split, args.modelCheckPoint)
    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset, forecast=args.forecast, forecast_mode=args.forecast_mode, classname=args.classname,
                                        rerank=args.rerank, tp_pct=args.tp_pct, root=args.root, static_only=args.static_only, cohort_analysis=args.cohort_analysis,
                                        K=args.K, C=args.C, split=args.split, version=args.version, eval_only=args.eval_only, jitter=args.jitter, 
                                        association_oracle=args.association_oracle, postprocess=args.postprocess, nogroup=args.nogroup)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
