import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

timesteps = 7
DOUBLE_FLIP=False
TWO_STAGE=False
REVERSE=False 
SPARSE=False
DENSE=True
BEV_MAP=False
FORECAST_FEATS=False
CLASSIFY=False
WIDE=False

sampler_type = "standard"

tasks = [
    dict(num_class=1, class_names=["car"]),
    #dict(num_class=2, class_names=["pedestrian"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        # type='SimpleVoxel',
        num_input_features=5,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)},
        share_conv_channel=64,
        dcn_head=False,
        timesteps=timesteps,
        two_stage=TWO_STAGE,
        reverse=REVERSE,
        sparse=SPARSE,
        dense=DENSE,
        bev_map=BEV_MAP,
        forecast_feature=FORECAST_FEATS,
        forecast_feature=FORECAST_FEATS,
        classify=CLASSIFY,
        wide_head=WIDE,
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=1000,
    min_radius=2,
    radius_mult = True,
    sampler_type=sampler_type,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-54, -54],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.075, 0.075],
    double_flip=DOUBLE_FLIP
)

# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 20
data_root = "/home/ubuntu/Workspace/Data/nuScenes/trainval_forecast"

if sampler_type == "standard":
    sample_group=[
        dict(car=2),
        #dict(pedestrian=2),
    ]
else:
    sample_group=[
        dict(static_car=2),
        #dict(static_pedestrian=2),
        dict(linear_car=4),
        #dict(linear_pedestrian=2),
        dict(nonlinear_car=6),
        #dict(nonlinear_pedestrian=4),
    ]

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path= data_root + "/dbinfos_train_20sweeps_withvelo.pkl",
    sample_groups=sample_group,
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                #pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
    sampler_type=sampler_type
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.9, 1.1],
    global_translate_std=0.5,
    db_sampler=db_sampler,
    class_names=class_names,
    sampler_type=sampler_type,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    sampler_type=sampler_type
)

voxel_generator = dict(
    range=[-54, -54, -5.0, 54, 54, 3.0],
    voxel_size=[0.075, 0.075, 0.2],
    max_points_in_voxel=10,
    max_voxel_num=[120000, 160000],
    double_flip=DOUBLE_FLIP
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
	dict(type="DoubleFlip") if DOUBLE_FLIP else dict(type="Empty"), 
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat", double_flip=DOUBLE_FLIP),
]

train_anno = data_root + "/infos_train_20sweeps_withvelo_filter_True.pkl"
val_anno = data_root + "/infos_val_20sweeps_withvelo_filter_True.pkl"
test_anno = data_root + "/infos_test_20sweeps_withvelo_filter_True.pkl"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        timesteps=timesteps,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        timesteps=timesteps,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        test_mode=True,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        version='v1.0-test',
        timesteps=timesteps,
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=25,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './models/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]
