import sys

sys.path.append('/home/nperi/Workspace/CenterForecast')
sys.path.append('/home/nperi/Workspace/Core/nuscenes-forecast/python-sdk')
import pickle
import json
import random
import operator
import numpy as np
import torch 
import pdb
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.constants import DETECTION_NAMES
    from nuscenes.utils.data_classes import Box
except:
    print("nuScenes devkit not found!")

from det3d.datasets.custom import PointCloudDataset
from det3d.core import box_torch_ops
from det3d.core.utils.circle_nms_jit import circle_nms
from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)
from pyquaternion import Quaternion
from det3d.datasets.registry import DATASETS

from tqdm import tqdm 


def get_sample_data(pred):
    box_list = [] 
    score_list = [] 
    pred = pred.copy() 

    for item in pred:    
        box =  Box(item['translation'], item['size'], Quaternion(item['rotation']),
                name=item['detection_name'])
        score_list.append(item['detection_score'])
        box_list.append(box)

    top_boxes = reorganize_boxes(box_list)
    top_scores = np.array(score_list).reshape(-1)

    return top_boxes, top_scores

def reorganize_boxes(box_lidar_nusc):
    rots = []
    centers = []
    wlhs = []
    for i, box_lidar in enumerate(box_lidar_nusc):
        v = np.dot(box_lidar.rotation_matrix, np.array([1, 0, 0]))
        rot = np.arctan2(v[1], v[0])

        rots.append(-rot- np.pi / 2)
        centers.append(box_lidar.center)
        wlhs.append(box_lidar.wlh)

    rots = np.asarray(rots)
    centers = np.asarray(centers)
    wlhs = np.asarray(wlhs)
    gt_boxes_lidar = np.concatenate([centers.reshape(-1,3), wlhs.reshape(-1,3), rots[..., np.newaxis].reshape(-1,1) ], axis=1)
    
    return gt_boxes_lidar

def reorganize_pred_by_class(pred):
    ret_dicts = defaultdict(list)
    for item in pred: 
        ret_dicts[item['detection_name']].append(item) 

    return ret_dicts

def group_classes(pred):
    ret_dicts = {"object" : []}
    for item in pred: 
        ret_dicts["object"].append(item) 

    return ret_dicts

def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep  

def forecast_nms(predictions, group_by="none"):
    res = {}
    for sample_token, prediction in tqdm(predictions.items()):
        annos = []
        # reorganize pred by class 

        if group_by == "class":
            pred_dicts = reorganize_pred_by_class(prediction)
        elif group_by == "none":
            pred_dicts = group_classes(prediction)
        else:
            assert False, "Invalid group_by {}".format(group_by)

        for name, pred in pred_dicts.items():
            # in global coordinate 
            top_boxes, top_scores = get_sample_data(pred)

            with torch.no_grad():
                top_boxes_tensor = torch.from_numpy(top_boxes)
                boxes_for_nms = top_boxes_tensor[:, [0, 1, 2, 4, 3, 5, -1]]
                boxes_for_nms[:, -1] = boxes_for_nms[:, -1] + np.pi /2 
                top_scores_tensor = torch.from_numpy(top_scores)

                selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms.float().cuda(), top_scores_tensor.float().cuda(), 
                                    thresh=0.2,
                                    pre_maxsize=None,
                                    post_max_size=50)
        
                #centers = boxes_for_nms[:, [0, 1]] 
                #boxes = torch.cat([centers, top_scores_tensor.float().view(-1, 1)], dim=1)
                #selected = _circle_nms(boxes, min_radius=1, post_max_size=50)  
            
            pred = [pred[s] for s in selected]

            annos.extend(pred)

        res.update({sample_token: annos})

    return res 

def get_token(scene_data, sample_data, sample_data_tokens, src_data_token, offset):
    scene = sample_data[sample_data_tokens.index(src_data_token)]["scene_token"]
    timestep = scene_data[scene].index(src_data_token) + offset

    if timestep > len(scene_data[scene]) - 1:
        timestep = len(scene_data[scene]) - 1

    if timestep < 0:
        timestep = 0

    dst_data_token = sample_data[sample_data_tokens.index(scene_data[scene][timestep])]["token"]

    return dst_data_token

def get_time(nusc, src_token, dst_token):
    time_last = 1e-6 * nusc.get('sample', src_token)["timestamp"]
    time_first = 1e-6 * nusc.get('sample', dst_token)["timestamp"]
    time_diff = time_first - time_last

    return time_diff 

def update_boxes(prev_boxes, curr_boxes, time_diff):
    for pbox, cbox in zip(prev_boxes, curr_boxes):
        cbox.center = pbox.center + time_diff * pbox.velocity      

    return curr_boxes


def box_center(boxes):
    center_box = np.array([box.center.tolist() + box.wlh.tolist() for box in boxes]) 
    return center_box

def distance_matrix(A, B, squared=False):
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared

def match_boxes(ret_boxes):
    box_centers = [box_center(boxes) for boxes in ret_boxes]
    cbox = box_centers[0]
    match_boxes, idx = [], []

    for fbox in box_centers:
        idx.append(np.argmin(distance_matrix(cbox, fbox), axis=1))

    for box, match in zip(ret_boxes, idx):
        match_boxes.append(np.array(box)[match])

    return match_boxes

def forecast_boxes(nusc, sample_data, scene_data, sample_data_tokens, det_forecast, forecast):
    fret_boxes, fret_tokens, rret_tokens = [], [], []
    
    det = deepcopy(det_forecast[0])
    boxes = _second_det_to_nusc_box(det)
    boxes = _lidar_nusc_box_to_global(nusc, boxes, det["metadata"]["token"])

    fret_boxes.append(boxes)
    fret_tokens.append(det["metadata"]["token"])
    rret_tokens.append(det["metadata"]["token"])

    if forecast > 0:
        for t in range(1, forecast + 1):
            
            if t > len(det_forecast) - 1:
                det = deepcopy(det_forecast[-1])
            else:
                det = deepcopy(det_forecast[t])
                
            boxes = _second_det_to_nusc_box(det)
            src_token, dst_token = fret_tokens[-1], get_token(scene_data, sample_data, sample_data_tokens, fret_tokens[-1], 1)
            rdst_token = get_token(scene_data, sample_data, sample_data_tokens, rret_tokens[-1], -1)            
            
            boxes = _lidar_nusc_box_to_global(nusc, boxes, src_token) 
            fret_boxes.append(boxes), fret_tokens.append(dst_token), rret_tokens.append(rdst_token)

        fret_boxes = match_boxes(fret_boxes)

    return fret_boxes, fret_tokens, rret_tokens

@DATASETS.register_module
class NuScenesDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, ring_index

    def __init__(
        self,
        info_path,
        root_path,
        nsweeps=0, # here set to zero to catch unset nsweep
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        version="v1.0-trainval",
        **kwargs,
    ):
        super(NuScenesDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"
        print(self.nsweeps)

        self._info_path = info_path
        self._class_names = class_names

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        self._num_point_features = NuScenesDataset.NumPointFeatures
        self._name_mapping = general_to_detection

        self.painted = kwargs.get('painted', False)
        if self.painted:
            self._num_point_features += 10 

        self.version = version
        self.eval_version = "detection_cvpr_2019"
        self.timesteps = kwargs.get("timesteps", None)
        
    def reset(self):
        self.logger.info(f"re-sample {self.frac} frames from full set")
        random.shuffle(self._nusc_infos_all)
        self._nusc_infos = self._nusc_infos_all[: self.frac]

    def load_infos(self, info_path):
        with open(self._info_path, "rb") as f:
            _nusc_infos_all = pickle.load(f)

        if not self.test_mode:  # if training
            self.frac = int(len(_nusc_infos_all) * 0.25)
            
            _cls_infos = {name: [] for name in self._class_names}
            for info in _nusc_infos_all:
                if len(info["gt_names"]) > 0:
                    for name in set(info["gt_names"][:,0]):
                        if name in self._class_names:
                            _cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {k: len(v) / max(duplicated_samples, 1) for k, v in _cls_infos.items()}

            self._nusc_infos = []

            frac = 1.0 / len(self._class_names)
            ratios = [frac / v for v in _cls_dist.values()]

            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                select = np.random.choice(np.array(range(len(cls_infos))), int(len(cls_infos) * ratio))
                self._nusc_infos += np.array(cls_infos)[select].tolist()

            _cls_infos = {name: [] for name in self._class_names}
            for info in self._nusc_infos:
                for name in set(info["gt_names"][:,0]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            _cls_dist = {
                k: len(v) / len(self._nusc_infos) for k, v in _cls_infos.items()
            }
        else:
            if isinstance(_nusc_infos_all, dict):
                self._nusc_infos = []
                for v in _nusc_infos_all.values():
                    self._nusc_infos.extend(v)
            else:
                self._nusc_infos = _nusc_infos_all

    def __len__(self):

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        return len(self._nusc_infos)

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._nusc_infos[0]:
            return None
        cls_range_map = config_factory(self.eval_version).serialize()['class_range']
        gt_annos = []
        for info in self._nusc_infos:
            try:
                gt_names = np.array(info["gt_names"][:,0])
                gt_boxes = info["gt_boxes"][:,0,:]
            except:
                gt_names = np.array(info["gt_names"])
                gt_boxes = info["gt_boxes"]

            mask = np.array([n != "ignore" for n in gt_names], dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            # det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = np.array([cls_range_map[n] for n in gt_names])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            N = int(np.sum(mask))
            gt_annos.append(
                {
                    "bbox": np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                    "alpha": np.full(N, -10),
                    "occluded": np.zeros(N),
                    "truncated": np.zeros(N),
                    "name": gt_names[mask],
                    "location": gt_boxes[mask][:, :3],
                    "dimensions": gt_boxes[mask][:, 3:6],
                    "rotation_y": gt_boxes[mask][:, 6],
                    "token": info["token"],
                }
            )
        return gt_annos

    def get_sensor_data(self, idx):
        info = self._nusc_infos[idx]

        res = { 
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": self.nsweeps,
                # "ground_plane": -gp[-1] if with_gp else None,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
                "timesteps" : self.timesteps
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "painted": self.painted 
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, output_dir=None, testset=False, forecast=6, tp_pct=0.6, root="/ssd0/nperi/nuScenes"):
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        if not testset:
            dets = []
            gt_annos = self.ground_truth_annotations
            assert gt_annos is not None

            miss = 0
            for gt in gt_annos:
                try:
                    dets.append(detections[gt["token"]])
                except Exception:
                    miss += 1

            assert miss == 0
        else:
            dets = [v for _, v in detections.items()]
            assert len(detections) == 6008

        nusc_annos = {
            "results": {},
            "meta": None,
        }

        nusc = NuScenes(version=version, dataroot=root, verbose=True)

        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        sample_data = [s for s in nusc.sample]
        sample_data_tokens = [s["token"] for s in nusc.sample]
        scene_tokens = [s["scene_token"] for s in nusc.sample]

        scene_data = {}

        for sample_tokens, scene_token in zip(sample_data_tokens, scene_tokens):
            if scene_token not in scene_data.keys():
                scene_data[scene_token] = []

            scene_data[scene_token].append(sample_tokens)

        for det_forecast in tqdm(dets):
            det_boxes, tokens, rtokens = forecast_boxes(nusc, sample_data, scene_data, sample_data_tokens, det_forecast, forecast)
            boxes, token = det_boxes[0], tokens[0]
            annos = []

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = None
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = None
                
                nusc_anno = {
                    "sample_token": token,
                    "forecast_sample_tokens" : tokens,
                    "reverse_sample_tokens" : rtokens,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "forecast_rotation" : [det_boxes[t][i].orientation.elements.tolist() for t in range(forecast)],
                    "rrotation": box.rorientation.elements.tolist(),
                    "forecast_rrotation" : [det_boxes[t][i].rorientation.elements.tolist() for t in range(forecast)],
                    "velocity": box.velocity[:2].tolist(),
                    "forecast_velocity" : [det_boxes[t][i].velocity[:2].tolist() for t in range(forecast)],
                    "rvelocity": box.rvelocity[:2].tolist(),
                    "forecast_rvelocity" : [det_boxes[t][i].rvelocity[:2].tolist() for t in range(forecast)],
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr
                    if attr is not None
                    else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                        0
                    ],
                }

                annos.append(nusc_anno)

            if token not in nusc_annos["results"].keys():
                nusc_annos["results"][token] = []

            nusc_annos["results"][token] += annos

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        name = self._info_path.split("/")[-1].split(".")[0]
        res_path = str(Path(output_dir) / Path(name + ".json"))
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            eval_main(
                nusc,
                self.eval_version,
                res_path,
                eval_set_map[self.version],
                output_dir,
                forecast=forecast,
                tp_pct=tp_pct
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            res_nusc = {
                "results": {"nusc": result},
                "detail": {"nusc": detail},
            }
        else:
            res_nusc = None

        if res_nusc is not None:
            res = {
                "results": {"nusc": res_nusc["results"]["nusc"],},
                "detail": {"eval.nusc": res_nusc["detail"]["nusc"],},
            }
        else:
            res = None

        return res, None
