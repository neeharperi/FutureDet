import os
import sys
import argparse
import pdb

sys.path.append('~/Workspace/FutureDet')
sys.path.append('~/Workspace/Core/nuscenes-forecast/python-sdk')

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tqdm import tqdm
from copy import deepcopy
from itertools import tee 
import pickle 

from nuscenes.eval.common.config import config_factory as detect_configs
from nuscenes.eval.common.loaders import (add_center_dist, filter_eval_boxes,
                                          load_gt, load_prediction)
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionConfig
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import (BoxVisibility, box_in_image,
                                           view_points)
from nuscenes.eval.detection.render import visualize_sample_forecast
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.data_classes import EvalBox
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from shapely.geometry import Polygon

def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
 
    return np.linalg.norm(np.array(pred_box["translation"][:2]) - np.array(gt_box["translation"][:2]))

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


def box2d_iou(boxA, boxB): 
    A = Box(center=boxA["translation"], size=boxA["size"], orientation=Quaternion(boxA["rotation"]))
    B = Box(center=boxB["translation"], size=boxB["size"], orientation=Quaternion(boxB["rotation"]))

    cornersA = view_points(A.corners(), np.eye(4), normalize=False)[:2, :].T
    cornersB = view_points(B.corners(), np.eye(4), normalize=False)[:2, :].T

    polyA = Polygon([(cornersA[0][0], cornersA[0][1]), (cornersA[1][0], cornersA[1][1]), (cornersA[5][0], cornersA[5][1]), (cornersA[4][0], cornersA[4][1])])
    polyB = Polygon([(cornersB[0][0], cornersB[0][1]), (cornersB[1][0], cornersB[1][1]), (cornersB[5][0], cornersB[5][1]), (cornersB[4][0], cornersB[4][1])])

    iou = polyA.intersection(polyB).area / polyA.union(polyB).area

    return iou 


def trajectory(nusc, box: DetectionBox) -> float:
    target = box.forecast_boxes[-1]
    time = [get_time(nusc, token[0], token[1]) for token in window([b["sample_token"] for b in box.forecast_boxes], 2)]

    static_forecast = deepcopy(box.forecast_boxes[0])

    if box2d_iou(target, static_forecast) > 0:
        return "static"

    linear_forecast = deepcopy(box.forecast_boxes[0])
    vel = linear_forecast["velocity"][:2]

    disp = np.sum(list(map(lambda x: np.array(list(vel) + [0]) * x, time)), axis=0)
    linear_forecast["translation"] = linear_forecast["translation"] + disp
    
    if box2d_iou(target, linear_forecast) > 0:
        return "linear"

    return "nonlinear"

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default="FutureDetection")
parser.add_argument('--model', default="forecast_n3dtf")
parser.add_argument('--forecast', type=int, default=7)
parser.add_argument('--architecture', default="centerpoint")
parser.add_argument('--classname', default="car")
parser.add_argument('--dataset', default="nusc")
parser.add_argument('--rootDirectory', default="~/Workspace/Data/nuScenes/")
parser.add_argument('--outputDirectory', default="Visuals/")

args = parser.parse_args()


experiment = args.experiment
model = args.model
forecast = args.forecast
architecture = args.architecture
classname = args.classname
dataset = args.dataset

rootDirectory = args.rootDirectory
outputDirectory = args.outputDirectory


det_dir = "models/{experiment}/{dataset}_{architecture}_{model}_detection/infos_val_20sweeps_withvelo_filter_True.json".format(architecture=architecture,
                                                                                   experiment=experiment,
                                                                                   model=model,
                                                                                   dataset=dataset)

if not os.path.isdir("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection".format(outputDirectory=outputDirectory, experiment=experiment, dataset=dataset, architecture=architecture, model=model)):
    os.makedirs("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection".format(outputDirectory=outputDirectory, experiment=experiment, dataset=dataset, architecture=architecture, model=model), exist_ok=True)


cfg = detect_configs("detection_forecast")

if os.path.isfile(rootDirectory + "/nusc.pkl"):
    nusc = pickle.load(open(rootDirectory + "/nusc.pkl", "rb"))
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=rootDirectory, verbose=True)
    pickle.dump(nusc, open(rootDirectory + "/nusc.pkl", "wb"))

pred_boxes, meta = load_prediction(det_dir, cfg.max_boxes_per_sample, DetectionBox, verbose=True)

if os.path.isfile(rootDirectory + "/gt.pkl"):
    gt_boxes = pickle.load(open(rootDirectory + "/gt.pkl", "rb"))
else:
    gt_boxes = load_gt(nusc, "val", DetectionBox, verbose=True, forecast=forecast)
    pickle.dump(gt_boxes, open(rootDirectory + "/gt.pkl", "wb"))

scenes = {}
for sample_token in tqdm(gt_boxes.boxes.keys()):
    gt = gt_boxes.boxes[sample_token]
    gt = [box for box in gt if box.detection_name == classname]

    if len(gt) == 0:
        continue
    
    pred = pred_boxes.boxes[sample_token]    
    pred_confs = [box.forecast_score for box in pred]
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]
    color = len(pred) * [("r", "r", "r")]

    taken = set()  # Initially no gt bounding box is matched.
    fid = set()
    for ind in sortind:
        pred_box = pred[ind].forecast_boxes[0]

        if pred[ind].forecast_id in fid:
            color[ind] = ("c", "c", "c")
            continue 

        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt):

            # Find closest match among ground truth boxes
            if not gt_idx in taken:
                this_distance = center_distance(gt_box.forecast_boxes[0], pred_box)
            
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < 1

        if is_match:
            taken.add(match_gt_idx)
            fid.add(pred[ind].forecast_id)
            color[ind] = ("b", "b", "b")

    pred_traj = []
    for box in pred:
        label = trajectory(nusc, box)
        pred_traj.append(label)

    gt_traj = []
    for box in gt:
        label = trajectory(nusc, box)
        gt_traj.append(label)

    visualize_sample_forecast(nusc, sample_token, gt, pred, gt_traj, pred_traj, color, savepath="{}".format("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection/{sample_token}.png".format(outputDirectory=outputDirectory,
                                                                                                                                                                                                experiment=experiment, 
                                                                                                                                                                                                dataset=dataset,
                                                                                                                                                                                                architecture=architecture, 
                                                                                                                                                                                                model=model,
                                                                                                                                                                                                sample_token=sample_token)))

    scene_token = nusc.get('sample', sample_token)['scene_token']
    
    if scene_token not in scenes:
        scenes[scene_token] = []

    scenes[scene_token].append(sample_token)

for scene_token in tqdm(scenes.keys()):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoOutput = cv2.VideoWriter("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection/{scene_token}.mp4".format(outputDirectory=outputDirectory, 
                                                                                                                                       experiment=experiment, 
                                                                                                                                       dataset=dataset, 
                                                                                                                                       architecture=architecture, 
                                                                                                                                       model=model, 
                                                                                                                                       scene_token=scene_token), fourcc, 2.0, (900, 900))

    for sample_token in scenes[scene_token]:
        img = cv2.imread("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection/{sample_token}.png".format(outputDirectory=outputDirectory, 
                                                                                                                                       experiment=experiment, 
                                                                                                                                       dataset=dataset, 
                                                                                                                                       architecture=architecture, 
                                                                                                                                       model=model, 
                                                                                                                                       sample_token=sample_token))
        videoOutput.write(img)
    
    videoOutput.release()