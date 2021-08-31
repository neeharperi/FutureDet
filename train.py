import os
import argparse
import pynvml 
import sys
import pdb 

sys.path.append('/home/nperi/Workspace/CenterForecast')
sys.path.append('/home/nperi/Workspace/Core/nuscenes-forecast/python-sdk')

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--experiment', required=True)
parser.add_argument('--debug', action="store_true")
parser.add_argument("--dataset", default="nusc")
parser.add_argument("--architecture", default="centerpoint")
parser.set_defaults(debug=False)
args = parser.parse_args()

model = args.model
experiment = args.experiment
dataset = args.dataset
architecture = args.architecture
configPath = "{dataset}_{architecture}_{model}_detection.py".format(dataset=dataset,
                                                                    architecture=architecture,
                                                                    model=model)

try:
    numDevices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
except:
    pynvml.nvmlInit()
    numDevices = pynvml.nvmlDeviceGetCount()

if args.debug:
    print("Starting in Debug Mode")
    os.system("python  ./tools/train.py configs/{architecture}/{configPath} --seed 0 --work_dir models/{experiment}/{dataset}_{architecture}_{model}_detection".format(architecture=architecture,
                                                                                                                                                              configPath=configPath,
                                                                                                                                                              experiment=experiment,
                                                                                                                                                              model=model,
                                                                                                                                                              dataset=dataset))
else:
    os.system("python -m torch.distributed.launch --nproc_per_node={numDevices} ./tools/train.py configs/{architecture}/{configPath} --work_dir models/{experiment}/{dataset}_{architecture}_{model}_detection".format(architecture=architecture,
                                                                                                                                                                                                                       configPath=configPath,
                                                                                                                                                                                                                       experiment=experiment,
                                                                                                                                                                                                                       model=model,
                                                                                                                                                                                                                       dataset=dataset,
                                                                                                                                                                                                                       numDevices=numDevices))

