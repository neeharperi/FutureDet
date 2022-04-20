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

parser = argparse.ArgumentParser()
parser.add_argument('--forecast', type=int, default=7)
parser.add_argument('--classname', default="car")
parser.add_argument('--rootDirectory', default="~/Workspace/Data/nuScenes/")

args = parser.parse_args()

forecast = args.forecast
rootDirectory = args.rootDirectory
classname = args.classname


cfg = detect_configs("detection_forecast")
nusc = NuScenes(version='v1.0-trainval', dataroot=rootDirectory, verbose=True)

gt_boxes = load_gt(nusc, "train", DetectionBox, verbose=True, forecast=forecast)

trajectory = []
for sample_token in tqdm(gt_boxes.boxes.keys()):
    boxes = gt_boxes.boxes[sample_token]

    for box in boxes:
        if box.detection_name != classname:
            continue 

        translation = box.translation
        velocity = box.velocity[:2]
        rotation = box.rotation
        position = [(velocity, rotation)] + [box.forecast_boxes[i]["translation"] - translation for i in range(1, forecast)]

        trajectory.append(position)


pickle.dump(trajectory, open("{}_trajectory.pkl".format(classname), "wb"))