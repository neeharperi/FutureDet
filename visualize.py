import os
import sys
import argparse
import pdb

sys.path.append('/home/nperi/Workspace/CenterForecast')
sys.path.append('/home/nperi/Workspace/Core/nuscenes-forecast/python-sdk')

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

from pyquaternion import Quaternion

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

def forecast_boxes(nusc, boxes):
    forecast_tokens = boxes.forecast_sample_tokens
    forecast_timediff = [get_time(nusc, token[0], token[1]) for token in window(forecast_tokens, 2)]
    forecast_rotation = boxes.forecast_rotation
    forecast_velocity = boxes.forecast_velocity
    forecast_boxes = [boxes]

    for i in range(len(forecast_tokens) - 1):
        new_box = deepcopy(forecast_boxes[-1])
        new_box.translation = new_box.translation + forecast_timediff[i] * np.append(forecast_velocity[i], 0)
        new_box.rotation = forecast_rotation[i]
        forecast_boxes.append(new_box)

    return forecast_boxes


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default="Forecast")
parser.add_argument('--model', default="forecast_n0")
parser.add_argument('--forecast', type=int, default=6)
parser.add_argument('--architecture', default="centerpoint")
parser.add_argument('--dataset', default="nusc")
parser.add_argument('--rootDirectory', default="/data/nperi/nuScenes/")
parser.add_argument('--outputDirectory', default="Visuals/")
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


cfg = detect_configs("detection_cvpr_2019")
nusc = NuScenes(version="v1.0-trainval", dataroot=rootDirectory, verbose=True)
pred_boxes, meta = load_prediction(det_dir, cfg.max_boxes_per_sample, DetectionBox, verbose=True)
gt_boxes = load_gt(nusc, "val", DetectionBox, verbose=True)

assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), "Samples in split don't match samples in predicted tracks."

pred_boxes = add_center_dist(nusc, pred_boxes)
gt_boxes = add_center_dist(nusc, gt_boxes)

pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, verbose=True)
gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=True)

scenes = {}
classname = ["car"]
for sample_token in tqdm(gt_boxes.boxes.keys()):
    gt = gt_boxes.boxes[sample_token]
    pred = pred_boxes.boxes[sample_token]
    
    gt_forecast = [forecast_boxes(nusc, box) for box in gt]
    pred_forecast = [forecast_boxes(nusc, box)  for box in pred]

    visualize_sample_forecast(nusc, sample_token, gt_forecast, pred_forecast, classname=classname, savepath="{}".format("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection/{sample_token}.png".format(outputDirectory=outputDirectory,
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