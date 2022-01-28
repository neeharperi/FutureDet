import numpy as np
import cv2
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES
import pdb

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = [v[i][inds[i]] for i in range(len(inds))]

def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

def forecast_augmentation(output):
    ground_truth, points = [], []

    for out in output:
        gt, pt = out

        ground_truth.append(gt), points.append(pt)

    return ground_truth, points

def forecast_voxelization(output):
    voxels, coordinates, num_points = [], [], []
    for out in output:
        vo, co, pt = out

        voxels.append(vo), coordinates.append(co), num_points.append(pt)

    return voxels, coordinates, num_points

def z_offset(points, 
                  meters_max=54,
                  pixels_per_meter=2,
                  hist_max_per_pixel=50,
                  zbins=np.array([-3.,   0.0, 1., 2.,  3., 10.]),
                  hist_normalize=True):
    assert(points.shape[-1] >= 3)
    assert(points.shape[0] > points.shape[1])
    meters_total = meters_max * 2
    pixels_total = meters_total * pixels_per_meter
    xbins = np.linspace(-meters_max, meters_max, pixels_total + 1, endpoint=True)
    ybins = xbins
    # The first left bin edge must match the last right bin edge.
    assert(np.isclose(xbins[0], -1 * xbins[-1]))
    assert(np.isclose(ybins[0], -1 * ybins[-1]))
    
    hist = np.histogramdd(points[..., :3], bins=(xbins, ybins, zbins), normed=False)[0]

    # Clip histogram 
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel

    # Normalize histogram by the maximum number of points in a bin we care about.
    if hist_normalize:
        overhead_splat = hist / hist_max_per_pixel
    else:
        overhead_splat = hist

    overhead_splat = cv2.resize(overhead_splat, dsize=(180, 180), interpolation=cv2.INTER_CUBIC)
    return overhead_splat, xbins, ybins, zbins

def get_mask(mask, t, angle, flip, scale):
    angle = np.degrees(-angle)
    if flip[0]:
        mask = np.fliplr(mask)
    if flip[1]:
        mask = np.flipud(mask)

    rot_mat = cv2.getRotationMatrix2D((90, 90), angle, scale)
    mask = cv2.warpAffine(mask, rot_mat, (180, 180))

    M = np.float32([[1, 0, -t[0]],
                    [0, 1, -t[1]]])

    mask = cv2.warpAffine(mask, M, (180, 180))

    return mask

@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_std = cfg.get('global_translate_std', 0)
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None 
                
            self.npoints = cfg.get("npoints", -1)
            self.sampler_type = cfg.sampler_type

        self.no_augmentation = cfg.get('no_augmentation', False)

    def __call__(self, res, info):
        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        elif res["type"] in ["NuScenesDataset"]:
            points = res["lidar"]["combined"]
        else:
            raise NotImplementedError

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": [np.array(box).reshape(-1) for box in anno_dict["names"]],
                "gt_trajectory": [np.array(box).reshape(-1) for box in anno_dict["trajectory"]],
            }

        if self.mode == "train" and not self.no_augmentation:
            selected = [drop_arrays_by_name(box, ["DontCare", "ignore", "UNKNOWN"]) for box in gt_dict["gt_names"]]
            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = [box_np_ops.points_count_rbbox(points, gt_dict["gt_boxes"][0]) for i in range(len(selected))]
                mask = [point_counts[i] >= self.min_points_in_gt for i in range(len(selected))]
                _dict_select(gt_dict, mask)

            gt_boxes_mask = [np.array([n in self.class_names for n in gt_dict["gt_names"][i]], dtype=np.bool_) for i in range(len(selected))]

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"][0],
                    gt_dict["gt_names"][0],
                    gt_dict["gt_trajectory"][0],
                    res["metadata"]["num_point_features"],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None,
                    sampler_type=self.sampler_type
                )
                
                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_trajectory = sampled_dict["gt_trajectory"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]

                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    
                    for i in range(len(gt_dict["gt_boxes"])):
                        for j in range(len(sampled_gt_boxes)):
                            try:
                                sampled_gt_boxes[j][-6:] = sampled_dict["gt_forecast"][j][i]
                            except:
                                sampled_gt_boxes[j][-6:] = sampled_dict["gt_forecast"][j][0]

                        gt_dict["gt_names"][i] = np.concatenate([gt_dict["gt_names"][i], sampled_gt_names], axis=0)
                        gt_dict["gt_trajectory"][i] = np.concatenate([gt_dict["gt_trajectory"][i], sampled_gt_trajectory], axis=0)

                        gt_dict["gt_boxes"][i] = np.concatenate([gt_dict["gt_boxes"][i], sampled_gt_boxes])
                        gt_boxes_mask[i] = np.concatenate([gt_boxes_mask[i], sampled_gt_masks], axis=0)

                    points = np.concatenate([sampled_points, points], axis=0)
            
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = [np.array([self.class_names.index(n) + 1 for n in gt_dict["gt_names"][i]], dtype=np.int32,) for i in range(len(gt_dict["gt_boxes"]))]
            gt_dict["gt_classes"] = gt_classes
            
            gt_dict["gt_boxes"], points, flip_aug = prep.random_flip_both(gt_dict["gt_boxes"], points)
            gt_dict["gt_boxes"], points, rot_aug = prep.global_rotation(gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise)
            gt_dict["gt_boxes"], points, scale_aug = prep.global_scaling_v2(gt_dict["gt_boxes"], points, *self.global_scaling_noise)
            gt_dict["gt_boxes"], points, trans_aug = prep.global_translate_(gt_dict["gt_boxes"], points, noise_translate_std=self.global_translate_std) 

            bev_map, xbins, ybins, zbins, = z_offset(points)
            bev = get_mask(anno_dict["bev"], t=trans_aug, angle=rot_aug, flip=flip_aug, scale=scale_aug)

            bev = np.concatenate((bev_map, bev[...,None]), axis=-1)
            res["lidar"]["bev_map"] = bev.transpose(2, 0, 1)

        elif self.no_augmentation:
            gt_boxes_mask = [np.array([n in self.class_names for n in gt_dict["gt_names"][i]], dtype=np.bool_) for i in range(len(gt_dict["gt_names"]))]
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = [np.array([self.class_names.index(n) + 1 for n in gt_dict["gt_names"][i]], dtype=np.int32,) for i in range(len(gt_dict["gt_names"]))]
            gt_dict["gt_classes"] = gt_classes
        
        if self.shuffle_points:
            rng = np.random.default_rng(0)
            rng.shuffle(points)
        
        res["lidar"]["points"] = points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num, int) else cfg.max_voxel_num

        self.double_flip = cfg.get('double_flip', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = [prep.filter_gt_box_outside_range(gt_dict["gt_boxes"][0], bv_range) for i in range(len(gt_dict["gt_boxes"]))]
            _dict_select(gt_dict, mask)

            res["lidar"]["annotations"] = gt_dict
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]
        
        voxels, coordinates, num_points = self.voxel_generator.generate(res["lidar"]["points"], max_voxels=max_voxels)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )

        double_flip = self.double_flip and (res["mode"] != 'train')

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )             
        
        return res, info

def flatten(box):
    return np.concatenate(box, axis=0)

def merge_multi_group_label(gt_classes, num_classes_by_task): 
    num_task = len(gt_classes)
    flag = 0 

    for i in range(num_task):
        gt_classes[i] += flag 
        flag += num_classes_by_task[i]

    return flatten(gt_classes)

@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.radius_mult = assigner_cfg.radius_mult
        self.sampler_type = assigner_cfg.sampler_type

        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius

    def __call__(self, res, info):
        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]

        # Calculate output featuremap size
        grid_size = res["lidar"]["voxels"]["shape"] 
        pc_range = res["lidar"]["voxels"]["range"]
        voxel_size = res["lidar"]["voxels"]["size"]

        feature_map_size = grid_size[:2] // self.out_size_factor

        example_forecast = []
        if res["mode"] == "train":
            length = len(res["lidar"]["annotations"]["gt_boxes"])
        else:
            length = len(res["lidar"]["annotations"]["boxes"])

        classname = "car" if "car" in class_names_by_task[0][0] else "pedestrian"

        if classname == "car":
            trajectory_map = {"static_car": 1, "linear_car" : 2, "nonlinear_car" : 3}
            forecast_map = {"car_1" : 1, "car_2" : 2, "car_3" : 3, "car_4" : 4, "car_5" : 5, "car_6" : 6, "car_7" : 7}
        else:
            trajectory_map = {"static_pedestrian": 1, "linear_pedestrian" : 2, "nonlinear_pedestrian" : 3}
            forecast_map = {"pedestrian_1" : 1, "pedestrian_2" : 2, "pedestrian_3" : 3, "pedestrian_4" : 4, "pedestrian_5" : 5, "pedestrian_6" : 6, "pedestrian_7" : 7}
        
        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            gt_dict["gt_names_trajectory"], gt_dict["gt_names_forecast"], gt_dict["gt_classes_trajectory"], gt_dict["gt_classes_forecast"], gt_dict["gt_boxes_trajectory"], gt_dict["gt_boxes_forecast"] = [], [], [], [], [], []

            for i in range(length):
                class_names = gt_dict["gt_names"][i]
                trajectory_names = gt_dict["gt_trajectory"][i]
                boxes = gt_dict["gt_boxes"][i]

                name_trajectories, classes_trajectories, boxes_trajectories = [], [], []

                for name, trajectory, box in zip(class_names, trajectory_names, boxes):
                    name_trajectories.append("{}_{}".format(trajectory, name))
                    classes_trajectories.append(trajectory_map["{}_{}".format(trajectory, name)])
                    boxes_trajectories.append(box)
                
                gt_dict["gt_names_trajectory"].append(np.array(name_trajectories))
                gt_dict["gt_classes_trajectory"].append(np.array(classes_trajectories))
                gt_dict["gt_boxes_trajectory"].append(np.array(boxes_trajectories))

            name_forecast, classes_forecast, boxes_forecast = [], [], []
            for i in range(length):
                class_names = gt_dict["gt_names"][i]
                boxes = gt_dict["gt_boxes"][i]

                for name, box in zip(class_names, boxes):
                    name_forecast.append("{}_{}".format(name, i + 1))
                    classes_forecast.append(forecast_map["{}_{}".format(name, i + 1)])
                    boxes_forecast.append(box)

            gt_dict["gt_names_forecast"] = length * [np.array(name_forecast)]
            gt_dict["gt_classes_forecast"] = length * [np.array(classes_forecast)]
            gt_dict["gt_boxes_forecast"] = length * [np.array(boxes_forecast)]

            for i in range(length):
                example = {}
                # reorganize the gt_dict by tasks
                task_masks = []
                flag = 0
                for class_name in class_names_by_task:
                    task_masks.append(
                        [
                            np.where(
                                gt_dict["gt_classes"][i] == class_name.index(j) + 1 + flag
                            )
                            for j in class_name
                        ]
                    )
                    flag += len(class_name)

                task_boxes = []
                task_classes = []
                task_names = []
                flag2 = 0
                for idx, mask in enumerate(task_masks):
                    task_box = []
                    task_class = []
                    task_name = []
                    for m in mask:
                        task_box.append(gt_dict["gt_boxes"][i][m])
                        task_class.append(gt_dict["gt_classes"][i][m] - flag2)
                        task_name.append(gt_dict["gt_names"][i][m])
                    task_boxes.append(np.concatenate(task_box, axis=0))
                    task_classes.append(np.concatenate(task_class))
                    task_names.append(np.concatenate(task_name))
                    flag2 += len(mask)

                for task_box in task_boxes:
                    # limit rad to [-pi, pi]
                    task_box[:, -1] = box_np_ops.limit_period(
                        task_box[:, -1], offset=0.5, period=np.pi * 2
                    )
                    task_box[:, -2] = box_np_ops.limit_period(
                        task_box[:, -2], offset=0.5, period=np.pi * 2
                    )

                # print(gt_dict.keys())
                gt_dict["gt_classes"][i] = task_classes
                gt_dict["gt_names"][i] = task_names
                gt_dict["gt_boxes"][i] = task_boxes

                res["lidar"]["annotations"] = gt_dict

                draw_gaussian = draw_umich_gaussian

                hms, anno_boxs, inds, masks, cats = [], [], [], [], []

                for idx, task in enumerate(self.tasks):
                    hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                                dtype=np.float32)

                    if res['type'] == 'NuScenesDataset':
                        # [reg, hei, dim, vx, vy, rots, rotc]
                        anno_box = np.zeros((max_objs, 14), dtype=np.float32)
                    elif res['type'] == 'WaymoDataset':
                        anno_box = np.zeros((max_objs, 10), dtype=np.float32) 
                    else:
                        raise NotImplementedError("Only Support nuScene for Now!")

                    ind = np.zeros((max_objs), dtype=np.int64)
                    mask = np.zeros((max_objs), dtype=np.uint8)
                    cat = np.zeros((max_objs), dtype=np.int64)

                    num_objs = min(gt_dict['gt_boxes'][i][idx].shape[0], max_objs)  

                    for k in range(num_objs):
                        cls_id = gt_dict['gt_classes'][i][idx][k] - 1

                        w, l, h = gt_dict['gt_boxes'][i][idx][k][3], gt_dict['gt_boxes'][i][idx][k][4], \
                                gt_dict['gt_boxes'][i][idx][k][5]
                        w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                        if w > 0 and l > 0:
                            vel_norm = np.linalg.norm(gt_dict['gt_boxes'][i][idx][k][6:8])

                            if self.radius_mult:
                                mult = min(max(1, vel_norm * (1 + i) / 2), 4)
                            else:
                                mult = 1.0

                            radius = mult * gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                            radius = max(self._min_radius, int(radius))

                            # be really careful for the coordinate system of your box annotation. 
                            x, y, z = gt_dict['gt_boxes'][i][idx][k][0], gt_dict['gt_boxes'][i][idx][k][1], \
                                    gt_dict['gt_boxes'][i][idx][k][2]

                            coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                            (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                            ct = np.array(
                                [coor_x, coor_y], dtype=np.float32)  
                            ct_int = ct.astype(np.int32)

                            # throw out not in range objects to avoid out of array area when creating the heatmap
                            if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                                continue 
                            
                            draw_gaussian(hm[cls_id], ct, radius)

                            new_idx = k
                            x, y = ct_int[0], ct_int[1]

                            cat[new_idx] = cls_id
                            ind[new_idx] = y * feature_map_size[0] + x
                            mask[new_idx] = 1

                            if res['type'] == 'NuScenesDataset': 
                                vx, vy = gt_dict['gt_boxes'][i][idx][k][6:8]
                                rvx, rvy = gt_dict['gt_boxes'][i][idx][k][8:10]
                                rot = gt_dict['gt_boxes'][i][idx][k][10]
                                rrot = gt_dict['gt_boxes'][i][idx][k][11]

                                anno_box[new_idx] = np.concatenate(
                                    (ct - (x, y), z, np.log(gt_dict['gt_boxes'][i][idx][k][3:6]),
                                    np.array(vx), np.array(vy), np.array(rvx), np.array(rvy), np.sin(rot), np.cos(rot), np.sin(rrot), np.cos(rrot)), axis=None)
                            elif res['type'] == 'WaymoDataset':
                                vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                                rot = gt_dict['gt_boxes'][idx][k][-1]
                                anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                            else:
                                raise NotImplementedError("Only Support Waymo and nuScene for Now")
                    

                    hms.append(hm)
                    anno_boxs.append(anno_box)
                    masks.append(mask)
                    inds.append(ind)
                    cats.append(cat)

                # used for two stage code 
                boxes = flatten(gt_dict['gt_boxes'][i])
                classes = merge_multi_group_label(gt_dict['gt_classes'][i], num_classes_by_task)

                if res["type"] == "NuScenesDataset":
                    gt_boxes_and_cls = np.zeros((max_objs, 13), dtype=np.float32)
                elif res['type'] == "WaymoDataset":
                    gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
                else:
                    raise NotImplementedError()

                boxes_and_cls = np.concatenate((boxes, 
                    classes.reshape(-1, 1).astype(np.float32)), axis=1)
                num_obj = len(boxes_and_cls)
                assert num_obj <= max_objs, "{} is greater than {}".format(num_obj, max_objs)
                # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
                boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 10, 11, 6, 7, 8, 9, 12]]
                gt_boxes_and_cls[:num_obj] = boxes_and_cls
        
                example.update({'gt_boxes_and_cls': gt_boxes_and_cls})
                example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})

                ###############################################################################################                
                if self.sampler_type != "standard":
                    if classname == "car":
                        class_trajectory_names_by_task = [["static_car", "linear_car", "nonlinear_car"]]
                    else:
                        class_trajectory_names_by_task = [["static_pedestrian", "linear_pedestrian", "nonlinear_pedestrian"]]

                    num_classes_trajectory_by_task = [3]
                    task_masks = []
                    flag = 0
                    for class_name in class_trajectory_names_by_task:
                        task_masks.append(
                            [
                                np.where(
                                    gt_dict["gt_classes_trajectory"][i] == class_name.index(j) + 1 + flag
                                )
                                for j in class_name
                            ]
                        )
                        flag += len(class_name)

                    task_boxes = []
                    task_classes = []
                    task_names = []
                    flag2 = 0
                    for idx, mask in enumerate(task_masks):
                        task_box = []
                        task_class = []
                        task_name = []
                        for m in mask:
                            task_box.append(gt_dict["gt_boxes_trajectory"][i][m])
                            task_class.append(gt_dict["gt_classes_trajectory"][i][m] - flag2)
                            task_name.append(gt_dict["gt_names_trajectory"][i][m])
                        task_boxes.append(np.concatenate(task_box, axis=0))
                        task_classes.append(np.concatenate(task_class))
                        task_names.append(np.concatenate(task_name))
                        flag2 += len(mask)

                    
                    for task_box in task_boxes:
                        # limit rad to [-pi, pi]
                        task_box[:, -1] = box_np_ops.limit_period(
                            task_box[:, -1], offset=0.5, period=np.pi * 2
                        )
                        task_box[:, -2] = box_np_ops.limit_period(
                            task_box[:, -2], offset=0.5, period=np.pi * 2
                        )
                
                    # print(gt_dict.keys())
                    gt_dict["gt_classes_trajectory"][i] = task_classes
                    gt_dict["gt_names_trajectory"][i] = task_names
                    gt_dict["gt_boxes_trajectory"][i] = task_boxes

                    res["lidar"]["annotations"] = gt_dict

                    draw_gaussian = draw_umich_gaussian

                    hms, anno_boxs, inds, masks, cats = [], [], [], [], []

                    for idx, task in enumerate(self.tasks):
                        hm = np.zeros((len(class_trajectory_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                                    dtype=np.float32)

                        if res['type'] == 'NuScenesDataset':
                            # [reg, hei, dim, vx, vy, rots, rotc]
                            anno_box = np.zeros((max_objs, 14), dtype=np.float32)
                        elif res['type'] == 'WaymoDataset':
                            anno_box = np.zeros((max_objs, 10), dtype=np.float32) 
                        else:
                            raise NotImplementedError("Only Support nuScene for Now!")

                        ind = np.zeros((max_objs), dtype=np.int64)
                        mask = np.zeros((max_objs), dtype=np.uint8)
                        cat = np.zeros((max_objs), dtype=np.int64)

                        num_objs = min(gt_dict['gt_boxes_trajectory'][i][idx].shape[0], max_objs)  

                        for k in range(num_objs):
                            cls_id = gt_dict['gt_classes_trajectory'][i][idx][k] - 1

                            w, l, h = gt_dict['gt_boxes_trajectory'][i][idx][k][3], gt_dict['gt_boxes_trajectory'][i][idx][k][4], \
                                    gt_dict['gt_boxes_trajectory'][i][idx][k][5]
                            w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                            if w > 0 and l > 0:
                                vel_norm = np.linalg.norm(gt_dict['gt_boxes_trajectory'][i][idx][k][6:8])

                                if self.radius_mult:
                                    mult = min(max(1, vel_norm * (1 + i) / 2), 4)
                                else:
                                    mult = 1.0

                                radius = mult * gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                                radius = max(self._min_radius, int(radius))

                                # be really careful for the coordinate system of your box annotation. 
                                x, y, z = gt_dict['gt_boxes_trajectory'][i][idx][k][0], gt_dict['gt_boxes_trajectory'][i][idx][k][1], \
                                        gt_dict['gt_boxes_trajectory'][i][idx][k][2]

                                coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                                (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                                ct = np.array(
                                    [coor_x, coor_y], dtype=np.float32)  
                                ct_int = ct.astype(np.int32)

                                # throw out not in range objects to avoid out of array area when creating the heatmap
                                if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                                    continue 
                                
                                draw_gaussian(hm[cls_id], ct, radius)

                                new_idx = k
                                x, y = ct_int[0], ct_int[1]

                                cat[new_idx] = cls_id
                                ind[new_idx] = y * feature_map_size[0] + x
                                mask[new_idx] = 1

                                if res['type'] == 'NuScenesDataset': 
                                    vx, vy = gt_dict['gt_boxes_trajectory'][i][idx][k][6:8]
                                    rvx, rvy = gt_dict['gt_boxes_trajectory'][i][idx][k][8:10]
                                    rot = gt_dict['gt_boxes_trajectory'][i][idx][k][10]
                                    rrot = gt_dict['gt_boxes_trajectory'][i][idx][k][11]

                                    anno_box[new_idx] = np.concatenate(
                                        (ct - (x, y), z, np.log(gt_dict['gt_boxes_trajectory'][i][idx][k][3:6]),
                                        np.array(vx), np.array(vy), np.array(rvx), np.array(rvy), np.sin(rot), np.cos(rot), np.sin(rrot), np.cos(rrot)), axis=None)
                                elif res['type'] == 'WaymoDataset':
                                    vx, vy = gt_dict['gt_boxes_trajectory'][idx][k][6:8]
                                    rot = gt_dict['gt_boxes_trajectory'][idx][k][-1]
                                    anno_box[new_idx] = np.concatenate(
                                    (ct - (x, y), z, np.log(gt_dict['gt_boxes_trajectory'][idx][k][3:6]),
                                    np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                                else:
                                    raise NotImplementedError("Only Support Waymo and nuScene for Now")
                        

                        hms.append(hm)
                        anno_boxs.append(anno_box)
                        masks.append(mask)
                        inds.append(ind)
                        cats.append(cat)

                    # used for two stage code 
                    boxes = flatten(gt_dict['gt_boxes_trajectory'][i])
                    classes = merge_multi_group_label(gt_dict['gt_classes_trajectory'][i], num_classes_trajectory_by_task)

                    if res["type"] == "NuScenesDataset":
                        gt_boxes_and_cls = np.zeros((max_objs, 13), dtype=np.float32)
                    elif res['type'] == "WaymoDataset":
                        gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
                    else:
                        raise NotImplementedError()

                    boxes_and_cls = np.concatenate((boxes, 
                        classes.reshape(-1, 1).astype(np.float32)), axis=1)
                    num_obj = len(boxes_and_cls)
                    assert num_obj <= max_objs, "{} is greater than {}".format(num_obj, max_objs)
                    # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
                    boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 10, 11, 6, 7, 8, 9, 12]]
                    gt_boxes_and_cls[:num_obj] = boxes_and_cls

                    example.update({'gt_boxes_and_cls_trajectory': gt_boxes_and_cls})
                    example.update({'hm_trajectory': hms, 'anno_box_trajectory': anno_boxs, 'ind_trajectory': inds, 'mask_trajectory': masks, 'cat_trajectory': cats})
           
                ############################################################################################### 

                    if classname == "car":               
                        class_forecast_names_by_task = [["car_1", "car_2", "car_3", "car_4", "car_5", "car_6", "car_7"]]
                    else:
                        class_forecast_names_by_task = [["pedestrian_1", "pedestrian_2", "pedestrian_3", "pedestrian_4", "pedestrian_5", "pedestrian_6", "pedestrian_7"]]

                    num_classes_forecast_by_task = [7]
                    task_masks = []
                    flag = 0
                    for class_name in class_forecast_names_by_task:
                        task_masks.append(
                            [
                                np.where(
                                    gt_dict["gt_classes_forecast"][i] == class_name.index(j) + 1 + flag
                                )
                                for j in class_name
                            ]
                        )
                        flag += len(class_name)

                    task_boxes = []
                    task_classes = []
                    task_names = []
                    flag2 = 0
                    for idx, mask in enumerate(task_masks):
                        task_box = []
                        task_class = []
                        task_name = []
                        for m in mask:
                            task_box.append(gt_dict["gt_boxes_forecast"][i][m])
                            task_class.append(gt_dict["gt_classes_forecast"][i][m] - flag2)
                            task_name.append(gt_dict["gt_names_forecast"][i][m])
                        task_boxes.append(np.concatenate(task_box, axis=0))
                        task_classes.append(np.concatenate(task_class))
                        task_names.append(np.concatenate(task_name))
                        flag2 += len(mask)

                    for task_box in task_boxes:
                        # limit rad to [-pi, pi]
                        task_box[:, -1] = box_np_ops.limit_period(
                            task_box[:, -1], offset=0.5, period=np.pi * 2
                        )
                        task_box[:, -2] = box_np_ops.limit_period(
                            task_box[:, -2], offset=0.5, period=np.pi * 2
                        )

                    # print(gt_dict.keys())
                    gt_dict["gt_classes_forecast"][i] = task_classes
                    gt_dict["gt_names_forecast"][i] = task_names
                    gt_dict["gt_boxes_forecast"][i] = task_boxes

                    res["lidar"]["annotations"] = gt_dict

                    draw_gaussian = draw_umich_gaussian

                    hms, anno_boxs, inds, masks, cats = [], [], [], [], []

                    for idx, task in enumerate(self.tasks):
                        hm = np.zeros((len(class_forecast_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                                    dtype=np.float32)

                        if res['type'] == 'NuScenesDataset':
                            # [reg, hei, dim, vx, vy, rots, rotc]
                            anno_box = np.zeros((max_objs, 14), dtype=np.float32)
                        elif res['type'] == 'WaymoDataset':
                            anno_box = np.zeros((max_objs, 10), dtype=np.float32) 
                        else:
                            raise NotImplementedError("Only Support nuScene for Now!")

                        ind = np.zeros((max_objs), dtype=np.int64)
                        mask = np.zeros((max_objs), dtype=np.uint8)
                        cat = np.zeros((max_objs), dtype=np.int64)

                        num_objs = min(gt_dict['gt_boxes_forecast'][i][idx].shape[0], max_objs)  

                        for k in range(num_objs):
                            cls_id = gt_dict['gt_classes_forecast'][i][idx][k] - 1

                            w, l, h = gt_dict['gt_boxes_forecast'][i][idx][k][3], gt_dict['gt_boxes_forecast'][i][idx][k][4], \
                                    gt_dict['gt_boxes_forecast'][i][idx][k][5]
                            w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                            if w > 0 and l > 0:
                                vel_norm = np.linalg.norm(gt_dict['gt_boxes_forecast'][i][idx][k][6:8])

                                if self.radius_mult:
                                    mult = min(max(1, vel_norm * (1 + i) / 2), 4)
                                else:
                                    mult = 1.0

                                radius = mult * gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                                radius = max(self._min_radius, int(radius))

                                # be really careful for the coordinate system of your box annotation. 
                                x, y, z = gt_dict['gt_boxes_forecast'][i][idx][k][0], gt_dict['gt_boxes_forecast'][i][idx][k][1], \
                                        gt_dict['gt_boxes_forecast'][i][idx][k][2]

                                coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                                (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                                ct = np.array(
                                    [coor_x, coor_y], dtype=np.float32)  
                                ct_int = ct.astype(np.int32)

                                # throw out not in range objects to avoid out of array area when creating the heatmap
                                if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                                    continue 
                                
                                draw_gaussian(hm[cls_id], ct, radius)

                                new_idx = k
                                x, y = ct_int[0], ct_int[1]

                                cat[new_idx] = cls_id
                                ind[new_idx] = y * feature_map_size[0] + x
                                mask[new_idx] = 1

                                if res['type'] == 'NuScenesDataset': 
                                    vx, vy = gt_dict['gt_boxes_forecast'][i][idx][k][6:8]
                                    rvx, rvy = gt_dict['gt_boxes_forecast'][i][idx][k][8:10]
                                    rot = gt_dict['gt_boxes_forecast'][i][idx][k][10]
                                    rrot = gt_dict['gt_boxes_forecast'][i][idx][k][11]

                                    anno_box[new_idx] = np.concatenate(
                                        (ct - (x, y), z, np.log(gt_dict['gt_boxes_forecast'][i][idx][k][3:6]),
                                        np.array(vx), np.array(vy), np.array(rvx), np.array(rvy), np.sin(rot), np.cos(rot), np.sin(rrot), np.cos(rrot)), axis=None)
                                elif res['type'] == 'WaymoDataset':
                                    vx, vy = gt_dict['gt_boxes_forecast'][idx][k][6:8]
                                    rot = gt_dict['gt_boxes_forecast'][idx][k][-1]
                                    anno_box[new_idx] = np.concatenate(
                                    (ct - (x, y), z, np.log(gt_dict['gt_boxes_forecast'][idx][k][3:6]),
                                    np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                                else:
                                    raise NotImplementedError("Only Support Waymo and nuScene for Now")
                        

                        hms.append(hm)
                        anno_boxs.append(anno_box)
                        masks.append(mask)
                        inds.append(ind)
                        cats.append(cat)

                    # used for two stage code 
                    boxes = flatten(gt_dict['gt_boxes_forecast'][i])
                    classes = merge_multi_group_label(gt_dict['gt_classes_forecast'][i], num_classes_forecast_by_task)

                    if res["type"] == "NuScenesDataset":
                        gt_boxes_and_cls = np.zeros((max_objs, 13), dtype=np.float32)
                    elif res['type'] == "WaymoDataset":
                        gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
                    else:
                        raise NotImplementedError()

                    boxes_and_cls = np.concatenate((boxes, 
                        classes.reshape(-1, 1).astype(np.float32)), axis=1)
                    num_obj = len(boxes_and_cls)
                    assert num_obj <= max_objs, "{} is greater than {}".format(num_obj, max_objs)
                    # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
                    boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 10, 11, 6, 7, 8, 9, 12]]
                    gt_boxes_and_cls[:num_obj] = boxes_and_cls

                    example.update({'gt_boxes_and_cls_forecast': gt_boxes_and_cls})
                    example.update({'hm_forecast': hms, 'anno_box_forecast': anno_boxs, 'ind_forecast': inds, 'mask_forecast': masks, 'cat_forecast': cats})
                
                example_forecast.append(example)

        else:
            example_forecast = length * [{}]

        ex = {k : [] for k in example_forecast[0].keys()}
        for ef in example_forecast:
            for k in ef.keys():
                ex[k].append(ef[k])

        res["lidar"]["targets"] = ex
        return res, info