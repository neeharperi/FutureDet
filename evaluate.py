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

sys.path.append('/home/ubuntu/Workspace/CenterForecast')
sys.path.append('/home/ubuntu/Workspace/Core/nuscenes-forecast/python-sdk')


from nuscenes.eval.detection.constants import getDetectionNames
#from nuscenes.eval.tracking.constants import TRACKING_NAMES
from nuscenes.nuscenes import NuScenes

os.environ['MKL_THREADING_LAYER'] = 'GNU'
detection_metrics = {"trans_err" : "ATE",
                     "scale_err" : "ASE",
                     "orient_err" : "AOE",
                     "vel_err" : "AVE",
                     "attr_err" : "AAE",
                     "avg_disp_err" : "ADE",
                     "final_disp_err" : "FDE",
                     "miss_rate" : "MR",
                    # "reverse_avg_disp_err" : "RADE",
                    # "reverse_final_disp_err" : "RFDE",
                    # "reverse_miss_rate" : "RMR",
                    }

detection_dataFrame = { "CLASS" : [],
                        "mAP" : [],
                        "mAR" : [],
                        "mFAP" : [],
                        "mFAR" : [],
                        "mAAP" : [],
                        "mAAR" : [],
                        "ATE" : [],
                        "ASE" : [],
                        "AOE" : [],
                        "AVE" : [],
                        "AAE" : [],
                        "ADE" : [],
                        "FDE" : [],
                        "MR" : [],
                        "mAP_MR" : [],
#                        "RADE" : [],
#                        "RFDE" : [],
#                        "RMR" : []
                     }

#tracking_metrics = {"amota" : "AMOTA",
#                    "amotp" : "AMOTP",
#                    "motar" : "MOTAR",
#                    "mota" : "MOTA",
#                    "tp" : "TP",
#                    "fp" : "FP",
#                    "fn" : "FN",
#                    "ids" : "ID_SWITCH",
#                    "frag" : "FRAGMENTED"}

#tracking_dataFrame = { "CLASS" : [],
#                        "AMOTA" : [],
#                        "AMOTP" : [],
#                        "MOTAR" : [],
#                        "MOTA" : [],
#                        "TP" : [],
#                        "FP" : [],
#                        "FN" : [],
#                        "ID_SWITCH" : [],
#                        "FRAGMENTED" : []}

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
parser.add_argument("--version", default="v1.0-trainval") #
parser.add_argument("--split", default="val") #
parser.add_argument("--modelCheckPoint", default="latest.pth")
parser.add_argument("--forecast", default=7)
parser.add_argument("--tp_pct", default=0.6)
parser.add_argument("--static_only", action="store_true")
parser.add_argument("--forecast_mode", default="velocity_forward")
parser.add_argument("--cohort_analysis", action="store_true")
parser.add_argument("--nms", action="store_true")
parser.add_argument("--K", default=1)

args = parser.parse_args()

architecture = args.architecture
experiment = args.experiment
rootDirectory = args.rootDirectory
model = args.model
dataset = args.dataset
version = args.version
split = args.split
extractBox = args.extractBox
modelCheckPoint = args.modelCheckPoint
forecast = args.forecast
forecast_mode = args.forecast_mode
tp_pct = args.tp_pct
static_only = args.static_only
cohort_analysis = args.cohort_analysis
nms = args.nms
K = args.K

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

os.system("python ./tools/dist_test.py configs/{architecture}/{configPath} {extractBox} --work_dir {det_dir} --checkpoint {det_dir}/{modelCheckPoint} --forecast {forecast} --forecast_mode {forecast_mode} --tp_pct {tp_pct} {static_only} {cohort_analysis} {nms} --K {K} --split {split} --version {version} --root {rootDirectory}".format(architecture=architecture, 
                                                                                                                                                                                    configPath=configPath, 
                                                                                                                                                                                    extractBox= "--extractBox" if extractBox else "", 
                                                                                                                                                                                    det_dir=det_dir, 
                                                                                                                                                                                    modelCheckPoint=modelCheckPoint,
                                                                                                                                                                                    forecast=forecast,
                                                                                                                                                                                    forecast_mode=forecast_mode,
                                                                                                                                                                                    tp_pct=tp_pct,
                                                                                                                                                                                    K=K,
                                                                                                                                                                                    static_only= "--static_only" if static_only else "",
                                                                                                                                                                                    cohort_analysis= "--cohort_analysis" if cohort_analysis else "",
                                                                                                                                                                                    nms= "--nms" if nms else "",
                                                                                                                                                                                    split=split,
                                                                                                                                                                                    version=version,
                                                                                                                                                                                    rootDirectory=rootDirectory))      
#print("Evaluating Tracking Results")

#if not os.path.isdir(track_dir):
#    os.mkdir(track_dir)

#os.system("python tools/nusc_tracking/pub_test.py --work_dir {track_dir}/ --checkpoint {det_dir}/infos_val_10sweeps_withvelo_filter_True.json --root {rootDirectory}".format(track_dir=track_dir, 
#                                                                                                                                                                             det_dir=det_dir, 
#                                                                                                                                                                             rootDirectory=rootDirectory))
##########################################################################
logFile = json.load(open(det_dir + "/metrics_summary.json"))

detection_dataFrame["CLASS"] = detection_dataFrame["CLASS"] + getDetectionNames(cohort_analysis)

for classname in detection_dataFrame["CLASS"]:
    detection_dataFrame["mAP"].append(logFile["mean_dist_aps"][classname])
    detection_dataFrame["mAR"].append(logFile["mean_dist_ars"][classname])

    detection_dataFrame["mFAP"].append(logFile["mean_dist_faps"][classname])
    detection_dataFrame["mFAR"].append(logFile["mean_dist_fars"][classname])

    detection_dataFrame["mAAP"].append(logFile["mean_dist_aaps"][classname])
    detection_dataFrame["mAAR"].append(logFile["mean_dist_aars"][classname])

    detection_dataFrame["mAP_MR"].append(logFile["mean_dist_aps_mr"][classname])

classMetrics = logFile["label_tp_errors"]
for metric in detection_metrics.keys():
    for classname in detection_dataFrame["CLASS"]:
        detection_dataFrame[detection_metrics[metric]].append(classMetrics[classname][metric])

detection_dataFrame = pd.DataFrame.from_dict(detection_dataFrame)

if not os.path.isdir("results/" + experiment + "/" + model):
    os.makedirs("results/" + experiment + "/" + model)

filename = "results/{experiment}/{model}/{dataset}_{architecture}_{model}_{forecast}_{forecast_mode}_{cohort}{static_only}{nms}detection.csv".format(experiment=experiment, model=model, dataset=dataset, architecture=architecture, forecast="t{}".format(forecast), forecast_mode=forecast_mode, cohort="cohort_" if cohort_analysis else "", static_only = "static_" if static_only else "", nms = "nms_" if nms else "")
detection_dataFrame.to_csv(filename, index=False)

#########################################################################
#logFile = json.load(open("{track_dir}/metrics_summary.json".format(track_dir=track_dir)))

#tracking_dataFrame["CLASS"] = tracking_dataFrame["CLASS"] + TRACKING_NAMES

#classMetrics = logFile["label_metrics"]
#for metric in tracking_metrics.keys():
#    for classname in tracking_dataFrame["CLASS"]:
#        tracking_dataFrame[tracking_metrics[metric]].append(classMetrics[metric][classname])
#tracking_dataFrame = pd.DataFrame.from_dict(tracking_dataFrame)

#if not os.path.isdir("results/" + experiment + "/" + model):
#    os.mkdir("results/" + experiment + "/" + model)

#filename = "results/{experiment}/{model}/{dataset}_{architecture}_{model}_{forecast}_tracking.csv".format(experiment=experiment, model=model, dataset=dataset, architecture=architecture, forecast="t{}".format(forecast))
#tracking_dataFrame.to_csv(filename, index=False)
#########################################################################
