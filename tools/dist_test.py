import numpy as np
import argparse
import copy
import json
import os
import sys


sys.path.append('/home/nperi/Workspace/CenterForecast')
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

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)

def load_pred(root):
    with open(os.path.join(root, "prediction.pkl"), "rb") as f:
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
    parser.add_argument("--tp_pct", type=float, default=0.6)

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def merge_dict(orig, new):
    orig["box3d_lidar"] = torch.cat([orig["box3d_lidar"], new["box3d_lidar"]])
    orig["scores"] = torch.cat([orig["scores"], new["scores"]])
    orig["label_preds"] = torch.cat([orig["label_preds"], new["label_preds"]])

    return orig
    
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
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    if args.extractBox:
        nusc = NuScenes(version="v1.0-trainval", dataroot=args.root, verbose=False)
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
        mode = "val"

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

            for t, timestep in enumerate(outputs):
                for output in timestep:
                    token = output["metadata"]["token"]
                    for k, v in output.items():
                        if k not in ["metadata"]:
                            output[k] = v.to(cpu_device)
                    
                    if token not in detections:
                        detections[token] = []
                    
                    detections[token].append(output)
                    
                    # detections.update({token: output,})
                    if args.local_rank == 0 and t == 0:
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
        
        save_pred(predictions, args.work_dir)

    if args.local_rank != 0:
        return
    
    predictions = load_pred(args.work_dir)
    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset, forecast=args.forecast, tp_pct=args.tp_pct, root=args.root)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
