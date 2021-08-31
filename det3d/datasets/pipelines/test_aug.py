from det3d import torchie

from ..registry import PIPELINES
from .compose import Compose
import pdb

@PIPELINES.register_module
class DoubleFlip(object):
    def __init__(self):
        pass

    def __call__(self, res, info):
        # y flip
        points = res["lidar"]["points"].copy()

        for i in range(len(points)):
            points[i][:, 1] = -points[i][:, 1]

        res["lidar"]['yflip_points'] = points

        # x flip
        points = res["lidar"]["points"].copy()

        for i in range(len(points)):
            points[i][:, 0] = -points[i][:, 0]

        res["lidar"]['xflip_points'] = points

        # x y flip
        points = res["lidar"]["points"].copy()
        
        for i in range(len(points)):
            points[i][:, 0] = -points[i][:, 0]
            points[i][:, 1] = -points[i][:, 1]

        res["lidar"]["double_flip_points"] = points  

        return res, info 



