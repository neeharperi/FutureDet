from datetime import time
import numpy as np
import pickle
import pdb 
from pathlib import Path
from functools import reduce
from typing import List

from tqdm import tqdm
from pyquaternion import Quaternion

from itertools import tee
from copy import deepcopy

try:
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import Box
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.utils.geometry_utils import transform_matrix
except:
    print("nuScenes devkit not Found!")

import pdb 

general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "ignore",
    "vehicle.bicycle": "ignore",
    "vehicle.bus.bendy": "ignore",
    "vehicle.bus.rigid": "ignore",
    "vehicle.truck": "ignore",
    "vehicle.construction": "ignore",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "ignore",
    "movable_object.barrier": "ignore",
    "movable_object.trafficcone": "ignore",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

def get_sample_data(nusc, sample_data_token: str, selected_anntokens: List[str] = None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param selected_anntokens: If provided only return the selected annotation.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    else:
        cam_intrinsic = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic

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

def center_distance(gt_box, pred_box) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.center[:2]) - np.array(gt_box.center[:2]))

def trajectory(nusc, boxes, time, timesteps=7):
    target = boxes[-1]
    
    static_forecast = deepcopy(boxes[0])

    linear_forecast = deepcopy(boxes[0])
    vel = linear_forecast.velocity[:2]
    disp = np.sum(list(map(lambda x: np.array(list(vel) + [0]) * x, time)), axis=0)
    linear_forecast.center = linear_forecast.center + disp
    
    if center_distance(target, static_forecast) < max(target.wlh[0], target.wlh[1]):
        return "static"

    elif center_distance(target, linear_forecast) < max(target.wlh[0], target.wlh[1]):
        return "linear"

    else:
        return "nonlinear"

def get_annotations(nusc, annotations, ref_boxes, timesteps):
    forecast_annotations = []
    forecast_boxes = []   
    forecast_trajectory = []
    sample_tokens = [s["token"] for s in nusc.sample]

    for annotation, ref_box in zip(annotations, ref_boxes):
        tracklet_box = []
        tracklet_annotation = []
        tracklet_trajectory = []

        token = nusc.sample[sample_tokens.index(annotation["sample_token"])]["data"]["LIDAR_TOP"]
        sd_record = nusc.get("sample_data", token)
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

        pannotation = annotation
        for i in range(timesteps):
            box = Box(center = annotation["translation"],
                      size = ref_box.wlh,
                      orientation = Quaternion(annotation["rotation"]),
                      velocity = nusc.box_velocity(annotation["token"]),
                      name = annotation["category_name"],
                      token = annotation["token"])

            box.translate(-np.array(pose_record["translation"]))
            box.rotate(Quaternion(pose_record["rotation"]).inverse)

            #  Move box to sensor coord system
            box.translate(-np.array(cs_record["translation"]))
            box.rotate(Quaternion(cs_record["rotation"]).inverse)

            tracklet_box.append(box)
            tracklet_annotation.append(annotation)
            
            next_token = annotation["next"]
            prev_token = pannotation["prev"]

            if next_token != "":
                annotation = nusc.get("sample_annotation", next_token)
            
            if prev_token != "":
                pannotation = nusc.get("sample_annotation", prev_token)
        
        tokens = [b["sample_token"] for b in tracklet_annotation]
        time = [get_time(nusc, src, dst) for src, dst in window(tokens, 2)]
        tracklet_trajectory = trajectory(nusc, tracklet_box, time, timesteps)

        forecast_boxes.append(tracklet_box)
        forecast_annotations.append(tracklet_annotation)
        forecast_trajectory.append(timesteps*[tracklet_trajectory])

    return forecast_boxes, forecast_annotations, forecast_trajectory

nusc = NuScenes(version="v1.0-trainval", dataroot="~/Workspace/Data/nuScenes", verbose=True)

ref_chan = "LIDAR_TOP"  # The radar channel from which we track back n sweeps to aggregate the point cloud.
chan = "LIDAR_TOP"  # The reference channel of the current sample_rec that the point clouds are mapped to.

trajectories  = {}
for sample in tqdm(nusc.sample):
    """ Manual save info["sweeps"] """
    ref_sd_token = sample["data"][ref_chan]
    #ref_sd_rec = nusc.get("sample_data", ref_sd_token)
    #ref_cs_rec = nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
    #ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
    #ref_time = 1e-6 * ref_sd_rec["timestamp"]

    ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)
    annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
    forecast_boxes, forecast_annotations, forecast_trajectory = get_annotations(nusc, annotations, ref_boxes, 7)
    names = [np.array([general_to_detection[b.name] for b in boxes]) for boxes in forecast_boxes]

    for traj, name in zip(forecast_trajectory, names):
        traj = traj[0]
        name = name[0]
        if name == "ignore":
            continue

        traj_name = traj + "_" + name

        if traj_name not in trajectories:
            trajectories[traj_name] = 0

        trajectories[traj_name] += 1

pdb.set_trace()