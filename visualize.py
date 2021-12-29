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

def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2]))

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default="FutureDetection")
parser.add_argument('--model', default="forecast_n3r")
parser.add_argument('--forecast', type=int, default=7)
parser.add_argument('--architecture', default="centerpoint")
parser.add_argument('--dataset', default="nusc")
parser.add_argument('--rootDirectory', default="/home/ubuntu/Workspace/Data/nuScenes/")
parser.add_argument('--outputDirectory', default="Visuals/")
parser.add_argument("--dets_only", action="store_true")

args = parser.parse_args()


experiment = args.experiment
model = args.model
forecast = args.forecast
architecture = args.architecture
dataset = args.dataset
dets_only = args.dets_only

rootDirectory = args.rootDirectory
outputDirectory = args.outputDirectory


det_dir = "models/{experiment}/{dataset}_{architecture}_{model}_detection/infos_val_20sweeps_withvelo_filter_True.json".format(architecture=architecture,
                                                                                   experiment=experiment,
                                                                                   model=model,
                                                                                   dataset=dataset)

if not os.path.isdir("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection".format(outputDirectory=outputDirectory, experiment=experiment, dataset=dataset, architecture=architecture, model=model)):
    os.makedirs("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection".format(outputDirectory=outputDirectory, experiment=experiment, dataset=dataset, architecture=architecture, model=model), exist_ok=True)


cfg = detect_configs("detection_forecast_cohort")

if os.path.isfile(rootDirectory + "/nusc.pkl"):
    nusc = pickle.load(open(rootDirectory + "/nusc.pkl", "rb"))
else:
    nusc = NuScenes(version='v1.0-mini', dataroot=rootDirectory, verbose=True)
    pickle.dump(nusc, open(rootDirectory + "/nusc.pkl", "wb"))

pred_boxes, meta = load_prediction(det_dir, cfg.max_boxes_per_sample, DetectionBox, verbose=True)

if os.path.isfile(rootDirectory + "/gt.pkl"):
    gt_boxes = pickle.load(open(rootDirectory + "/gt.pkl", "rb"))
else:
    gt_boxes = load_gt(nusc, "mini_val", DetectionBox, verbose=True, forecast=forecast)
    pickle.dump(gt_boxes, open(rootDirectory + "/gt.pkl", "wb"))

classname = ["car"]

scenes = {}
for sample_token in tqdm(gt_boxes.boxes.keys()):
    gt = gt_boxes.boxes[sample_token]

    if sample_token not in pred_boxes.boxes.keys():
        continue

    pred = pred_boxes.boxes[sample_token]

    gt = [box for box in gt if box.detection_name in classname]

    if len(gt) == 0:
        continue

    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name in classname]
    pred_confs = [box.detection_score for box in pred_boxes_list]
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    match_count = 0
    match_pred, match_gt = [], []
    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
    
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_boxes_list[ind].sample_token]):

            # Find closest match among ground truth boxes
            if not (pred_boxes_list[ind].sample_token, gt_idx) in taken:
                this_distance = center_distance(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        is_match = min_dist < 0.5
        # If the closest match is close enough according to threshold we have a match!
        if is_match:
            taken.add((pred_boxes_list[ind].sample_token, match_gt_idx))
            match_count += 1
            mp = pred_boxes_list[ind]
            mg = gt_boxes[pred_boxes_list[ind].sample_token][match_gt_idx]

            match_pred.append(mp)
            match_gt.append(mg)

    visualize_sample_forecast(nusc, sample_token, match_gt, match_pred, classname=classname, dets_only=dets_only, savepath="{}".format("{outputDirectory}/{experiment}/{dataset}_{architecture}_{model}_detection/{sample_token}.png".format(outputDirectory=outputDirectory,
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