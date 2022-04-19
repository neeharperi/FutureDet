import sys

sys.path.append('/home/ubuntu/Workspace/CenterForecast')
sys.path.append('/home/ubuntu/Workspace/Core/nuscenes-forecast/python-sdk')

import pickle
import json
import random
import operator
import numpy as np
import os
import torch 
import cv2
import pdb
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from itertools import tee
from nuscenes.utils.geometry_utils import view_points
from shapely.geometry import Polygon

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.constants import getDetectionNames
    from nuscenes.utils.data_classes import Box
    from nuscenes.eval.detection.data_classes import  DetectionBox

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
import networkx as nx
import itertools
from tqdm import tqdm 


def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)

    return zip(*iters)

def get_time(nusc, src_token, dst_token):
    time_last = 1e-6 * nusc.get('sample', src_token)["timestamp"]
    time_first = 1e-6 * nusc.get('sample', dst_token)["timestamp"]
    time_diff = time_first - time_last

    return time_diff 

def get_token(scene_data, sample_data, sample_data_tokens, src_data_token, offset):
    scene = sample_data[sample_data_tokens.index(src_data_token)]["scene_token"]
    timestep = scene_data[scene].index(src_data_token) + offset

    if timestep > len(scene_data[scene]) - 1:
        timestep = len(scene_data[scene]) - 1

    if timestep < 0:
        timestep = 0

    dst_data_token = sample_data[sample_data_tokens.index(scene_data[scene][timestep])]["token"]

    return dst_data_token

def box_center(boxes):
    center_box = np.array([box.center[:2].tolist() for box in boxes]) 
    return center_box

def box_past_center(time, boxes):
    center_box = np.array([(box.center[:2] - time * box.velocity[:2]).tolist() for box in boxes]) 
    return center_box

def box_future_center(time, boxes):
    center_box = np.array([(box.center[:2] + time * box.velocity[:2]).tolist() for box in boxes]) 
    return center_box

def box_center_(boxes):
    center_box = np.array([box["translation"] for box in boxes]) 
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

def tracker(classname, time, ret_boxes):
    if classname == "car":
        reject_thresh = 2
        match_thresh = 0.25

    else:
        reject_thresh = 1
        match_thresh = 0.25
    
    reverse_time = time[::-1]
    reverse_ret_boxes = ret_boxes[::-1]
    trajectory = []
    
    ####################################################################
    if classname in ["car", "pedestrian"]:
        ## Forecasting 
        idx, dist = [], []
        for timesteps, tm in zip(window(ret_boxes, 2), time):
            current, future = timesteps
            
            curr = box_center(current)
            curr_future = box_future_center(tm, current)
            futr = box_center(future)
            
            if len(curr) == 0 or len(futr) == 0:
                continue 

            dist_mat = distance_matrix(curr_future, futr)
            min_idx = np.argmin(dist_mat, axis=1)
            min_dist = np.min(dist_mat, axis=1)
            idx.append(min_idx)
            dist.append(min_dist)
        
        if len(idx) != len(ret_boxes) - 1:
            return []

        trajectory_idxs = []
        for i in range(idx[0].shape[0]):
            trajectory_idx = [i]
            void = False
            for ind, dis in zip(idx, dist):
                if dis[trajectory_idx[-1]] > reject_thresh:
                    void = True 

                trajectory_idx.append(ind[trajectory_idx[-1]])

            if not void:
                trajectory_idxs.append(trajectory_idx)
            
        for idxs in trajectory_idxs:
            forecast = []
            for ind, boxes in zip(idxs, ret_boxes):
                forecast.append(boxes[ind])

            trajectory.append(forecast)
        
        ## Constant Velocity Forward
        for idx in np.arange(len(ret_boxes[0])):
            curr = ret_boxes[0][idx]
            velocity = curr.velocity

            forecast = [curr]
            for t in time:
                new_box = deepcopy(forecast[-1])
                new_box.center = new_box.center + t * velocity
                forecast.append(new_box)

            trajectory.append(forecast)
        ##########################################################################
        ## Back-Casting 
        idx, dist = [], []
        for timesteps, tm in zip(window(reverse_ret_boxes, 2), reverse_time):
            current, previous = timesteps
            
            curr = box_center(current)
            curr_past = box_past_center(tm, current)
            prev = box_center(previous)
            
            if len(curr) == 0 or len(prev) == 0:
                continue 
            
            dist_mat = distance_matrix(curr_past, prev)
            min_idx = np.argmin(dist_mat, axis=1)
            min_dist = np.min(dist_mat, axis=1)
            idx.append(min_idx)
            dist.append(min_dist)
        
        if len(idx) != len(ret_boxes) - 1:
            return []

        trajectory_idxs = []
        for i in range(idx[0].shape[0]):
            trajectory_idx = [i]
            void = False
            for ind, dis in zip(idx, dist):
                if dis[trajectory_idx[-1]] > reject_thresh:
                    void = True 

                trajectory_idx.append(ind[trajectory_idx[-1]])

            if not void:
                trajectory_idxs.append(trajectory_idx)
            
        for idxs in trajectory_idxs:
            forecast = []
            for ind, boxes in zip(idxs, reverse_ret_boxes):
                forecast.append(boxes[ind])

            forecast = forecast[::-1]
            trajectory.append(forecast)
        

        # Constant Velocity Backward
        #for idx in np.arange(len(ret_boxes[-1])):
        #    curr = deepcopy(ret_boxes[-1][idx])
        #    velocity = curr.velocity

        #    forecast = [curr]
        #    for t in time:
        #        new_box = deepcopy(forecast[-1])
        #        new_box.center = new_box.center - t * velocity
        #        forecast.append(new_box)

        #    forecast = forecast[::-1]

        #    anchor_center = box_center(ret_boxes[0])
        #    forecast_center = box_center([forecast[0]])
        #    dist = distance_matrix(forecast_center, anchor_center)
            
        #    if np.min(dist) < match_thresh:
        #        trajectory.append(forecast)
        
    return trajectory
    
def box_serialize(box, token, name, attr):
    ret = {"sample_token": token,
            "translation": box.center.tolist(),
            "size": box.wlh.tolist(),
            "rotation": box.orientation.elements.tolist(),
            "velocity": box.velocity[:2].tolist(),
            "detection_name": name,
            "detection_score": box.score,
            "forecast_score": box.score,
            "forecast_id": -1,
            "attribute_name": attr
            if attr is not None
            else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                0
                    ],
        }

    return ret 

def network_split(L):
    G=nx.from_edgelist(L)

    l=list(nx.connected_components(G))
    # after that we create the map dict , for get the unique id for each nodes
    mapdict={z:x for x, y in enumerate(l) for z in y }
    # then append the id back to original data for groupby 
    newlist=[ x+(mapdict[x[0]],)for  x in L]
    #using groupby make the same id into one sublist
    newlist=sorted(newlist,key=lambda x : x[2])
    yourlist=[list(y) for x , y in itertools.groupby(newlist,key=lambda x : x[2])]

    ret = {}

    for group in yourlist:
        for pair in group:
            a, b, i = pair
            ret[(a, b)] = i

    return ret

def multi_future(forecast_boxes, classname):
    match_thresh = 0.25

    for sample_token in forecast_boxes.keys():
        boxes = [box for box in forecast_boxes[sample_token] if classname in box["detection_name"]]
        pred_center = box_center_(boxes)
        if len(pred_center) == 0:
                    continue 

        dist_mat = distance_matrix(pred_center, pred_center)
        idxa, idxb = np.where(dist_mat < match_thresh)

        L = []
        for ida, idb in zip(idxa, idxb):
            L.append((ida, idb))

        net = network_split(L)
        for ida, idb in zip(idxa, idxb):
            forecast_id = net[(ida, idb)]
            boxes[ida]["forecast_id"] = forecast_id
            boxes[idb]["forecast_id"] = forecast_id

            detection_score = boxes[ida]["detection_score"]
            forecast_score = boxes[ida]["forecast_score"]

            for box in boxes[ida]["forecast_boxes"]:
                box["detection_score"] = detection_score
                box["forecast_score"] = forecast_score
                box["forecast_id"] = forecast_id

            detection_score = boxes[idb]["detection_score"]
            forecast_score = boxes[idb]["forecast_score"]

            for box in boxes[idb]["forecast_boxes"]:
                box["detection_score"] = detection_score
                box["forecast_score"] = forecast_score
                box["forecast_id"] = forecast_id
            
        forecast_boxes[sample_token] = boxes

    return forecast_boxes

def process_trajectories(nusc, sample_token, ret_boxes, forecast, train_dist):
    #sample_rec = nusc.get('sample', sample_token)
    #sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    #cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    #pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    #ego_map = nusc.get_ego_centric_map(sd_record["token"])
    #bev = cv2.resize(ego_map, dsize=(50, 50), interpolation=cv2.INTER_CUBIC).T

    test_trajectories = []
    for ret_box in ret_boxes:
        box = ret_box[0]

        translation = box.center
        velocity = box.velocity[:2]
        rotation = [box.orientation[0], box.orientation[1], box.orientation[2], box.orientation[3]]
        
        position = [(velocity, rotation)] + [ret_box[i].center - translation for i in range(1, forecast)]
        test_trajectories.append(position)

    test_dist = []
    for trajectory in test_trajectories:
        velocity, rotation = trajectory[0]
        test_dist.append(np.array(list(velocity) + rotation + list(np.hstack(trajectory[1:]))))

    test_dist = np.array(test_dist)

    dist = distance_matrix(train_dist, test_dist)
    idx = np.argmin(dist, axis=0)
    matched_trajectory = (train_dist[idx])

    out_boxes = [] 
    for i, out in enumerate(zip(ret_boxes, matched_trajectory)):
        ret_box, trajectory = out
        translation = deepcopy(ret_box[0].center)

        trajectory = trajectory[6:]
        for i in range(forecast - 1):
            ret_box[i + 1].center = translation + trajectory[3*i:3*i+3]

        out_boxes.append(deepcopy(ret_box))

    return out_boxes

def forecast_boxes(nusc, sample_data, scene_data, sample_data_tokens, det_forecast, forecast, forecast_mode, classname, jitter, K, C, train_dist, postprocess):
    ret_boxes, ret_tokens = [], []

    for t in range(forecast):
        dst_token =  get_token(scene_data, sample_data, sample_data_tokens, det_forecast["metadata"]["token"], t)
        ret_tokens.append(dst_token)

    time = []
    stale = False
    for src, dst in window(ret_tokens, 2):
        elapse_time = get_time(nusc, src, dst)

        if elapse_time == 0:
            stale = True

        time.append(elapse_time)

    for t in range(forecast):
        mask = np.array(det_forecast["label_preds"] == t)
        box3d = det_forecast["box3d_lidar"][mask]
        scores = det_forecast["scores"][mask]
        labels = det_forecast["label_preds"][mask]
        det = {"box3d_lidar" : box3d, "scores" : scores, "label_preds" : labels, "metadata" : det_forecast["metadata"]}

        boxes = _second_det_to_nusc_box(det)
        boxes = _lidar_nusc_box_to_global(nusc, boxes, det["metadata"]["token"])

        ret_boxes.append(boxes)

    if stale or len(ret_boxes[0]) == 0:
        return [], ret_tokens

    if forecast_mode in ["velocity_constant", "velocity_forward", "velocity_reverse"]:
        ret_boxes = match_boxes(ret_boxes)

    elif forecast_mode in ["velocity_sparse_forward", "velocity_sparse_reverse", "velocity_sparse_match"]:
        forward_box = [box[0] for box in ret_boxes]
        reverse_box = [box[1] for box in ret_boxes]

        forward_box = match_boxes(forward_box)
        reverse_box = match_boxes(reverse_box)

        ret_boxes = [[forward, reverse] for forward, reverse in zip(forward_box, reverse_box)]

    if "dense" not in forecast_mode: 
        trajectory_boxes = []
        for j in range(len(ret_boxes[0])):
            boxes = []      

            for i in range(forecast):
                boxes.append(ret_boxes[i][j])

            trajectory_boxes.append(boxes)
    else:
        forecast_boxes = tracker(classname, time, ret_boxes)

    if forecast_mode in ["velocity_constant", "velocity_forward", "velocity_reverse"]:
        if forecast_mode == "velocity_reverse":
            time = time[::-1]

        ret_boxes = []
        for trajectory_box in trajectory_boxes:
            forecast_boxes = [trajectory_box[0]]
            for i in range(forecast - 1):
                new_box = deepcopy(forecast_boxes[-1])

                if forecast_mode == "velocity_reverse":
                    new_box.center = new_box.center - time[i] * trajectory_box[i].velocity
                else:
                    new_box.center = new_box.center + time[i] * trajectory_box[i].velocity

                forecast_boxes.append(new_box)
                
            if forecast_mode == "velocity_reverse":
                forecast_boxes = forecast_boxes[::-1]

            ret_boxes.append(forecast_boxes)
        
    elif forecast_mode == "velocity_dense":
        ret_boxes = forecast_boxes 
        
        if postprocess:
            sample_token = nusc.get("sample", ret_tokens[0])["token"]
            ret_boxes = process_trajectories(nusc, sample_token, ret_boxes, forecast, train_dist)

    else:
        assert False, "Invalid Forecast Mode"


    if jitter:
        jitter_boxes = []
        for trajectory_box in ret_boxes:
            for _ in range(K - 1):
                start_box = trajectory_box[0]
                vel_norm = C * np.linalg.norm(start_box.velocity)
                start_vel = start_box.velocity
                jittered_vel = np.random.normal(start_vel, np.array([vel_norm, vel_norm, vel_norm]))

                forecast_boxes = [start_box]
                for i in range(forecast - 1):
                    new_box = deepcopy(forecast_boxes[-1])
                    new_box.center = new_box.center + time[i] * jittered_vel

                    forecast_boxes.append(new_box)

                jitter_boxes.append(forecast_boxes)
        
        ret_boxes = ret_boxes + jitter_boxes

    return ret_boxes, ret_tokens

def trajectory_score(fboxes, rerank, timesteps):
    if rerank == "first":
        return fboxes[0]["detection_score"]

    if rerank == "last":
        return fboxes[-1]["detection_score"]

    elif rerank == "add":
        return np.sum([fboxes[i]["detection_score"] for i in range(timesteps)]) / timesteps

    elif rerank == "mult":
        return np.product([fboxes[i]["detection_score"] for i in range(timesteps)])

    assert False, "{} is Invalid".format(rerank)


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
            try:
                det_range = np.array([cls_range_map[n] for n in gt_names])
            except:
                det_range = np.array([50 for n in gt_names])

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

        if "bev_map" in res["lidar"]:
            data["bev_map"] = res["lidar"]["bev_map"]
        
        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, output_dir=None, testset=False, forecast=7, forecast_mode="velocity_forward", classname="car", rerank="last", tp_pct=0.6, root="/ssd0/nperi/nuScenes", 
                   static_only=False, cohort_analysis=False, K=1, C=1, split="val", version="v1.0-trainval", eval_only=False, jitter=False, 
                   association_oracle=False, postprocess=False, nogroup=False):
        self.eval_version = "detection_forecast"
        name = self._info_path.split("/")[-1].split(".")[0]

        if postprocess:
            name = name + "_pp"
            
        res_path = str(Path(output_dir) / Path(name + ".json"))

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
        
        if os.path.isfile(root + "/nusc.pkl"):
            nusc = pickle.load(open(root + "/nusc.pkl", "rb"))
        else:
            nusc = NuScenes(version=version, dataroot=root, verbose=True)
            pickle.dump(nusc, open(root + "/nusc.pkl", "wb"))

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
        
        train_trajectories = pickle.load(open("/home/ubuntu/Workspace/CenterForecast/{}_trajectory.pkl".format(classname), "rb"))
        train_dist = []
        for trajectory in train_trajectories:
            velocity, rotation = trajectory[0]
            train_dist.append(np.array(list(velocity) + rotation + list(np.hstack(trajectory[1:]))))

        train_dist = np.array(train_dist)

        if not eval_only:
            for j, det_forecast in enumerate(tqdm(dets)):
                det_boxes, tokens = forecast_boxes(nusc, sample_data, scene_data, sample_data_tokens, det_forecast, forecast, forecast_mode, classname, jitter, K, C, train_dist, postprocess)
                token = tokens[0]
                annos = []
                
                for i, boxes in enumerate(det_boxes):
                    box = boxes[0]
                    name = classname

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

          
                    attr = attr if attr is not None else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]

                    fboxes = [box_serialize(box, token, name, attr) for box, token in zip(boxes, tokens)]
                    
                    forecast_score = trajectory_score(fboxes, rerank, forecast)

                    nusc_anno = {
                        "sample_token": token,
                        "translation": box.center.tolist(),
                        "size": box.wlh.tolist(),
                        "rotation": box.orientation.elements.tolist(),
                        "velocity": box.velocity[:2].tolist(),
                        "forecast_boxes" : fboxes,
                        "detection_name": name,
                        "detection_score": fboxes[0]["detection_score"],
                        "forecast_score" : forecast_score,
                        "forecast_id" : (i + 1) * (j + 1), 
                        "attribute_name": attr,
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
            
            if not nogroup:
                nusc_annos["results"] = multi_future(nusc_annos["results"], classname)

            with open(res_path, "w") as f:
                json.dump(nusc_annos, f)
        
        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            eval_main(
                nusc,        
                "detection_forecast_cohort" if cohort_analysis else "detection_forecast",
                res_path,
                split,
                output_dir,
                forecast=forecast,
                tp_pct=tp_pct,
                static_only=static_only,
                cohort_analysis=cohort_analysis,
                topK=K,
                root=root,
                association_oracle=association_oracle,
                nogroup=nogroup
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"

            for name in getDetectionNames(cohort_analysis):
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
