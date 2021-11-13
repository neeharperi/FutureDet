import os
import sys
import argparse
import pdb

sys.path.append('/home/ubuntu/Workspace/CenterForecast')
sys.path.append('/home/ubuntu/Workspace/Core/nuscenes-forecast/python-sdk')

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tqdm import tqdm
from copy import deepcopy
from itertools import tee 

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

def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box["translation"][:2]) - np.array(gt_box["translation"][:2]))

def trajectory(nusc, box: DetectionBox, timesteps=7) -> float:
    target = box.forecast_boxes[-1]
    static_forecast = box.forecast_boxes[0]
    
    if center_distance(target, static_forecast) < max(static_forecast["size"][0], static_forecast["size"][1]):
        return "static"
    else:
        return "moving"


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default="Forecast")
parser.add_argument('--model', default="forecast_n3d")
parser.add_argument('--forecast', type=int, default=7)
parser.add_argument('--architecture', default="centerpoint")
parser.add_argument('--dataset', default="nusc")
parser.add_argument('--rootDirectory', default="/home/ubuntu/Workspace/Data/nuScenes/")
parser.add_argument('--outputDirectory', default="mini-Visuals/")

args = parser.parse_args()


experiment = args.experiment
model = args.model
forecast = args.forecast
architecture = args.architecture
dataset = args.dataset

rootDirectory = args.rootDirectory
outputDirectory = args.outputDirectory


det_dir = "models/{experiment}/{dataset}_{architecture}_{model}_detection/infos_val_10sweeps_withvelo_filter_True.json".format(architecture=architecture,
                                                                                   experiment=experiment,
                                                                                   model=model,
                                                                                   dataset=dataset)

if not os.path.isdir("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection".format(outputDirectory=outputDirectory, experiment=experiment, dataset=dataset, architecture=architecture, model=model)):
    os.makedirs("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection".format(outputDirectory=outputDirectory, experiment=experiment, dataset=dataset, architecture=architecture, model=model), exist_ok=True)


cfg = detect_configs("detection_forecast_cohort")
nusc = NuScenes(version="v1.0-mini", dataroot=rootDirectory, verbose=True)
pred_boxes, meta = load_prediction(det_dir, cfg.max_boxes_per_sample, DetectionBox, verbose=True)
gt_boxes = load_gt(nusc, "mini_val", DetectionBox, verbose=True)

for sample_token in pred_boxes.boxes.keys():
    for box in pred_boxes.boxes[sample_token]:
        label = trajectory(nusc, box, forecast)
        box.detection_name = label + "_" + box.detection_name

for sample_token in gt_boxes.boxes.keys():
    for box in gt_boxes.boxes[sample_token]:
        label = trajectory(nusc, box, forecast)
        box.detection_name = label + "_" + box.detection_name

assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), "Samples in split don't match samples in predicted tracks."

pred_boxes = add_center_dist(nusc, pred_boxes)
gt_boxes = add_center_dist(nusc, gt_boxes)

pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, verbose=True)
gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=True)


scenes = {}
classname = ["moving_car"]
for sample_token in tqdm(gt_boxes.boxes.keys()):
    gt = gt_boxes.boxes[sample_token]
    pred = pred_boxes.boxes[sample_token]

    class_gt = [box for box in gt if box.detection_name in classname]

    if len(class_gt) == 0:
        continue

    visualize_sample_forecast(nusc, sample_token, gt, pred, classname=classname, savepath="{}".format("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection/{sample_token}.png".format(outputDirectory=outputDirectory,
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