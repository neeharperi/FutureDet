import torch
import numpy as np
import pdb 

original = torch.load("/home/nperi/Workspace/FutureDet/models/Forecast/nusc_centerpoint_forecast_n0_detection/latest.pth")["state_dict"]
frozen = torch.load("/home/nperi/Workspace/FutureDet/models/Test/nusc_centerpoint_forecast_n5_t_detection/latest.pth")["state_dict"]

for key in frozen.keys():
    try:
        t1 = original[key]
        t2 = frozen[key]
        same = torch.sum(t1 == t2)
        dims = np.prod(np.array(t1.shape))

        if same != dims:
            print(key)

    except:
        print(key)
