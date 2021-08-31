import argparse
import json
import os
import pdb
from copy import deepcopy
from pprint import pprint
import numpy as np
import mkl
import pandas as pd
import pynvml
import sys
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.eval.tracking.constants import TRACKING_NAMES
from nuscenes.nuscenes import NuScenes

sys.path.append('/home/nperi/Workspace/CenterForecast')
sys.path.append('/home/nperi/Workspace/Core/nuscenes-forecast/python-sdk')

os.environ['MKL_THREADING_LAYER'] = 'GNU'
detection_metrics = {"trans_err" : "ATE",
                     "scale_err" : "ASE",
                     "orient_err" : "AOE",
                     "vel_err" : "AVE",
                     "attr_err" : "AAE"}

detection_dataFrame = { "CLASS" : [],
                        "mAP" : [],
                        "ATE" : [],
                        "ASE" : [],
                        "AOE" : [],
                        "AVE" : [],
                        "AAE" : []}

tracking_metrics = {"amota" : "AMOTA",
                    "amotp" : "AMOTP",
                    "motar" : "MOTAR",
                    "mota" : "MOTA",
                    "tp" : "TP",
                    "fp" : "FP",
                    "fn" : "FN",
                    "ids" : "ID_SWITCH",
                    "frag" : "FRAGMENTED"}

tracking_dataFrame = { "CLASS" : [],
                        "AMOTA" : [],
                        "AMOTP" : [],
                        "MOTAR" : [],
                        "MOTA" : [],
                        "TP" : [],
                        "FP" : [],
                        "FN" : [],
                        "ID_SWITCH" : [],
                        "FRAGMENTED" : []}

try:
    numDevices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
except:
    pynvml.nvmlInit()
    numDevices = pynvml.nvmlDeviceGetCount()

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--experiment', required=True)
parser.add_argument("--rootDirectory", default="/home/ubuntu/Workspace/Data/nuScenes")
parser.add_argument("--dataset", default="nusc")
parser.add_argument('--architecture', default="centerpoint")
parser.add_argument("--extractBox", action="store_true")
parser.add_argument("--version", default="v1.0-trainval")
parser.add_argument("--modelCheckPoint", default="latest.pth")
parser.add_argument("--forecast", default=0)


args = parser.parse_args()

architecture = args.architecture
experiment = args.experiment
rootDirectory = args.rootDirectory
model = args.model
dataset = args.dataset
version = args.version
extractBox = args.extractBox
modelCheckPoint = args.modelCheckPoint
forecast = args.forecast

configPath = "{dataset}_{architecture}_{model}_detection.py".format(dataset=dataset,
                                                                    architecture=architecture,
                                                                    model=model)

det_dir = "models/{experiment}/{dataset}_{architecture}_{model}_detection".format(architecture=architecture,
                                                                                   experiment=experiment,
                                                                                   model=model,
                                                                                   dataset=dataset)

track_dir = "models/{experiment}/{dataset}_{architecture}_{model}_tracking".format(architecture=architecture,
                                                                                   experiment=experiment,
                                                                                   model=model,
                                                                                   dataset=dataset)
print("Evaluating Detection Results for " + modelCheckPoint)

os.system("python ./tools/dist_test.py configs/{architecture}/{configPath} {extractBox} --work_dir {det_dir} --checkpoint {det_dir}/{modelCheckPoint} --forecast {forecast}".format(architecture=architecture, 
                                                                                                                                                                                    configPath=configPath, 
                                                                                                                                                                                    extractBox= "--extractBox" if extractBox else "", 
                                                                                                                                                                                    det_dir=det_dir, 
                                                                                                                                                                                    modelCheckPoint=modelCheckPoint,
                                                                                                                                                                                    forecast=forecast))      
print("Evaluating Tracking Results")

if not os.path.isdir(track_dir):
    os.mkdir(track_dir)

os.system("python tools/nusc_tracking/pub_test.py --work_dir {track_dir}/ --checkpoint {det_dir}/infos_val_10sweeps_withvelo_filter_True.json --root {rootDirectory}".format(track_dir=track_dir, 
                                                                                                                                                                             det_dir=det_dir, 
                                                                                                                                                                             rootDirectory=rootDirectory))
##########################################################################
logFile = json.load(open(det_dir + "/metrics_summary.json"))

detection_dataFrame["CLASS"] = detection_dataFrame["CLASS"] + DETECTION_NAMES

for classname in detection_dataFrame["CLASS"]:
    detection_dataFrame["mAP"].append(logFile["mean_dist_aps"][classname])
   
classMetrics = logFile["label_tp_errors"]
for metric in detection_metrics.keys():
    for classname in detection_dataFrame["CLASS"]:
        detection_dataFrame[detection_metrics[metric]].append(classMetrics[classname][metric])

detection_dataFrame = pd.DataFrame.from_dict(detection_dataFrame)

if not os.path.isdir("results/" + experiment + "/" + model):
    os.makedirs("results/" + experiment + "/" + model)

filename = "results/{experiment}/{model}/{dataset}_{architecture}_{model}_{forecast}_detection.csv".format(experiment=experiment, model=model, dataset=dataset, architecture=architecture, forecast="t{}".format(forecast))
detection_dataFrame.to_csv(filename, index=False)
#########################################################################
logFile = json.load(open("{track_dir}/metrics_summary.json".format(track_dir=track_dir)))

tracking_dataFrame["CLASS"] = tracking_dataFrame["CLASS"] + TRACKING_NAMES

classMetrics = logFile["label_metrics"]
for metric in tracking_metrics.keys():
    for classname in tracking_dataFrame["CLASS"]:
        tracking_dataFrame[tracking_metrics[metric]].append(classMetrics[metric][classname])
tracking_dataFrame = pd.DataFrame.from_dict(tracking_dataFrame)

if not os.path.isdir("results/" + experiment + "/" + model):
    os.mkdir("results/" + experiment + "/" + model)

filename = "results/{experiment}/{model}/{dataset}_{architecture}_{model}_{forecast}_tracking.csv".format(experiment=experiment, model=model, dataset=dataset, architecture=architecture, forecast="t{}".format(forecast))
tracking_dataFrame.to_csv(filename, index=False)
#########################################################################