# ------------------------------------------------------------------------------
# Portions of this code are from
# det3d (https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)
# Copyright (c) 2019 朱本金
# Licensed under the MIT License
# ------------------------------------------------------------------------------

import logging
from collections import defaultdict
from posixpath import join

from tqdm.utils import RE_ANSI
from det3d.core import box_torch_ops
import torch
from det3d.torchie.cnn import kaiming_init
from torch import double, nn
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss, ForecastLoss
from det3d.models.utils import Sequential
from ..registry import HEADS
import copy 
try:
    from det3d.ops.dcn import DeformConv
except:
    print("Deformable Convolution not built!")

from det3d.core.utils.circle_nms_jit import circle_nms
import pdb 

class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            in_channels, deformable_groups * offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()

    def forward(self, x,):
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x

class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        two_stage=False,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads 
        self.two_stage = two_stage

        if self.two_stage:
            if "vel" in self.heads and "rot" in self.heads:
                self.forecast_conv = nn.Sequential(
                    nn.Conv2d(head_conv, head_conv,
                    kernel_size=3, padding=1, bias=True),
                    nn.BatchNorm2d(head_conv),
                    nn.ReLU(inplace=True)
                )

            if "rvel" in self.heads and "rrot" in self.heads: 
                self.reverse_conv = nn.Sequential(
                    nn.Conv2d(in_channels, head_conv,
                    kernel_size=3, padding=1, bias=True),
                    nn.BatchNorm2d(head_conv),
                    nn.ReLU(inplace=True)
                )

        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv-1):
                fc.add(nn.Conv2d(in_channels, head_conv,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())

            fc.add(nn.Conv2d(head_conv, classes,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))    

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)
        

    def forward(self, x):
        ret_dict = dict()        
        for head in self.heads:

            if self.two_stage:
                if head in ["vel", "rot"]:
                    shared_forecast = self.forecast_conv(x)
                    ret_dict[head] = self.__getattr__(head)(shared_forecast)

                if head in ["rvel", "rrot"]:
                    shared_reverse = self.reverse_conv(x)
                    ret_dict[head] = self.__getattr__(head)(shared_reverse)

            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict

class DCNSepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_cls,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(DCNSepHead, self).__init__(**kwargs)

        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4) 
        
        self.feature_adapt_reg = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4)  

        # heatmap prediction head 
        self.cls_head = Sequential(
            nn.Conv2d(in_channels, head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_cls,
                kernel_size=3, stride=1, 
                padding=1, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(init_bias)

        # other regression target 
        self.task_head = SepHead(in_channels, heads, head_conv=head_conv, bn=bn, final_kernel=final_kernel)


    def forward(self, x):    
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret['hm'] = cls_score

        return ret


@HEADS.register_module
class CenterHead(nn.Module):
    def __init__(
        self,
        in_channels=[128,],
        tasks=[],
        dataset='nuscenes',
        weight=0.25,
        code_weights=[],
        common_heads=dict(),
        logger=None,
        init_bias=-2.19,
        share_conv_channel=64,
        num_hm_conv=2,
        dcn_head=False,
        timesteps=1,
        two_stage=False,
        reverse=False,
        sparse=False,
        dense=False,
        bev_map=False,
    ):
        super(CenterHead, self).__init__()
        
        self.two_stage = two_stage 
        self.reverse = reverse
        self.sparse = sparse
        self.dense = dense
        self.bev_map = bev_map
        self.target_timesteps = 7

        if not self.reverse and not self.sparse and not self.dense:
            self.standard = True
        else:
            self.standard = False 

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights
        self.box_n_dim = 7  

        if 'vel' in common_heads and 'rvel' in common_heads and 'rot' in common_heads and 'rrot' in common_heads:
            self.box_n_dim = 13
            self.code_weights_forecast = list(np.array(self.code_weights) * np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]))
            
            self.code_weights_two_stage_forecast = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

        elif 'vel' in common_heads and 'rot' in common_heads:
            self.box_n_dim = 9
            self.code_weights_forecast = list(np.array(self.code_weights) * np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0]))

            self.code_weights_two_stage_forecast = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


        self.weight = weight  # weight between hm loss and loc loss
        self.dataset = dataset

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()
        self.crit_forecast = ForecastLoss()
        
        self.use_direction_classifier = False 

        self.timesteps = timesteps
        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(
            f"num_classes: {num_classes}"
        )

        # a shared convolution 

        if self.bev_map:
            self.shared_conv = nn.Sequential(
                nn.Conv2d(in_channels + 5, share_conv_channel,
                kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(share_conv_channel),
                nn.ReLU(inplace=True)
            )
        else:
            self.shared_conv = nn.Sequential(
                nn.Conv2d(in_channels, share_conv_channel,
                kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(share_conv_channel),
                nn.ReLU(inplace=True)
            )
        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        if dcn_head:
            print("Use Deformable Convolution in the CenterHead!")

        if self.sparse:
            #self.num_classes = 2 * [1, 1]
            self.num_classes = 2 * [1]

        if self.dense:
            #self.num_classes = self.timesteps * [1, 1]
            self.num_classes = self.timesteps * [1]

        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)

            for head in heads.keys():
                if not self.dense and head in ["vel", "rvel"]:
                    heads[head] = (self.timesteps * heads[head][0], heads[head][1])
                
            if not dcn_head:
                heads.update(dict(hm=(num_cls, num_hm_conv)))

                self.tasks.append(
                    SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3, two_stage=self.two_stage)
                )
            else:
                self.tasks.append(
                    DCNSepHead(share_conv_channel, num_cls, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )
        logger.info("Finish CenterHead Initialization")

    def forward(self, x, bev_map=None, *kwargs):
        ret_dicts = []

        if self.bev_map:
            x = torch.cat([bev_map, x], axis=1)

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def loss(self, example, preds_dicts, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            hm = self.num_classes[task_id]
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            #HM Loss
            if self.two_stage:
                hm_loss = torch.tensor(0).cuda()
            elif self.reverse:
                hm_loss = self.crit(preds_dict['hm'], example['hm'][-1][task_id], example['ind'][-1][task_id], example['mask'][-1][task_id], example['cat'][-1][task_id])
            elif self.sparse:
                #hm_loss = self.crit(preds_dict['hm'], example['hm'][(self.timesteps - 1) * (task_id // 2)][task_id % 2], example['ind'][(self.timesteps - 1) * (task_id // 2)][task_id % 2], example['mask'][(self.timesteps - 1) * (task_id // 2)][task_id % 2], example['cat'][(self.timesteps - 1) * (task_id // 2)][task_id % 2])
                hm_loss = self.crit(preds_dict['hm'], example['hm'][(self.timesteps - 1) * (task_id)][0], example['ind'][(self.timesteps - 1) * (task_id)][0], example['mask'][(self.timesteps - 1) * (task_id)][0], example['cat'][(self.timesteps - 1) * (task_id)][0])
            elif self.dense:
                #hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id // 2][task_id % 2], example['ind'][task_id // 2][task_id % 2], example['mask'][task_id // 2][task_id % 2], example['cat'][task_id // 2][task_id % 2])
                hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id][0], example['ind'][task_id][0], example['mask'][task_id][0], example['cat'][task_id][0])
            else:
                hm_loss = self.crit(preds_dict['hm'], example['hm'][0][task_id], example['ind'][0][task_id], example['mask'][0][task_id], example['cat'][0][task_id])

            #Generate Target Boxes
            if self.reverse:
                target_box = [example['anno_box'][i][task_id] for i in range(self.timesteps)][::-1]
            elif self.sparse:
                if task_id == 0:
                    #target_box = [example['anno_box'][i][task_id % 2] for i in range(self.timesteps)]
                    target_box = [example['anno_box'][i][0] for i in range(self.timesteps)]
                else:
                    #target_box = [example['anno_box'][i][task_id % 2] for i in range(self.timesteps)][::-1]
                    target_box = [example['anno_box'][i][0] for i in range(self.timesteps)][::-1]
            elif self.dense:
                #target_box = [example['anno_box'][task_id // 2][task_id % 2] for i in range(self.timesteps)]
                target_box = [example['anno_box'][task_id][0] for i in range(self.timesteps)]
            else:
                target_box = [example['anno_box'][i][task_id] for i in range(self.timesteps)]

            # reconstruct the anno_box from multiple reg heads
            if self.dataset in ['waymo', 'nuscenes']:
                if 'vel' in preds_dict and 'rvel' in preds_dict and 'rot' in preds_dict and 'rrot' in preds_dict:
                    if self.dense:
                        preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                preds_dict['vel'], preds_dict['rvel'], preds_dict['rot'], preds_dict['rrot']), dim=1)
                    
                    else:
                        preds_dict['anno_box'] = [torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                        preds_dict['vel'][:,2*i:2*i+2,::], preds_dict['rvel'][:,2*i:2*i+2,::], preds_dict['rot'], preds_dict['rrot']), dim=1) for i in range(self.timesteps)]
                    
                elif 'vel' in preds_dict and 'rot' in preds_dict:
                    if self.dense:
                        preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                            preds_dict['vel'], preds_dict['rot']), dim=1)
            
                    else:
                        preds_dict['anno_box'] = [torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                            preds_dict['vel'][:,2*i:2*i+2,::], preds_dict['rot']), dim=1) for i in range(self.timesteps)]
            
                   
                    target_box = [target_box[i][..., [0, 1, 2, 3, 4, 5, 6, 7, -2, -1]] for i in range(self.timesteps)]# remove vel target                       

                else:
                    preds_dict['anno_box'] = [torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['rot']), dim=1) for i in range(self.timesteps)]
                    
                    target_box = [target_box[i][..., [0, 1, 2, 3, 4, 5, -2, -1]] for i in range(self.timesteps)]# remove vel target                       
            else:
                raise NotImplementedError()

            ret = {}

            # Regression loss for dimension, offset, height, rotation   
            if self.reverse:
                box_loss = [self.crit_reg(preds_dict['anno_box'][i], example['mask'][-1][task_id], example['ind'][-1][task_id], target_box[i]) for i in range(self.timesteps)]

            elif self.sparse:
                #box_loss = [self.crit_reg(preds_dict['anno_box'][i], example['mask'][(self.timesteps - 1) * (task_id // 2)][task_id % 2], example['ind'][(self.timesteps - 1) * (task_id // 2)][task_id % 2], target_box[(self.timesteps - 1) * (task_id // 2)]) for i in range(self.timesteps)]
                box_loss = [self.crit_reg(preds_dict['anno_box'][i], example['mask'][(self.timesteps - 1) * (task_id)][0], example['ind'][(self.timesteps - 1) * (task_id)][0], target_box[(self.timesteps - 1) * (task_id)]) for i in range(self.timesteps)]

            elif self.dense:
                #box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id // 2][task_id % 2], example['ind'][task_id // 2][task_id % 2], target_box[task_id // 2])
                box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id][0], example['ind'][task_id][0], target_box[task_id])

            else:
                box_loss = [self.crit_reg(preds_dict['anno_box'][i], example['mask'][0][task_id], example['ind'][0][task_id], target_box[i]) for i in range(self.timesteps)]

         
            loc_loss = []
            
            if self.two_stage:
                for i in range(self.timesteps):
                    loc_loss.append((box_loss[i] * box_loss[i].new_tensor(self.code_weights_two_stage_forecast)).sum())
            else:
                if self.dense:
                    for i in range(self.timesteps):
                        loc_loss.append((box_loss[i] * box_loss[i].new_tensor(self.code_weights)).sum())

                else:
                    for i in range(self.timesteps):
                        loc_loss.append((box_loss[i] * box_loss[i].new_tensor(self.code_weights)).sum() if i == 0 else (box_loss[i] * box_loss[i].new_tensor(self.code_weights_forecast)).sum())
            
            loss = hm_loss + self.weight * sum(loc_loss)

            if self.sparse:
                #ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss': loc_loss, 'loc_loss_elem': [box_loss[i].detach().cpu() for i in range(self.timesteps)], 'num_positive': sum(sum(example['mask'][(self.timesteps - 1) * (task_id // 2)][task_id % 2].float()))})
                ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss': loc_loss, 'loc_loss_elem': [box_loss[i].detach().cpu() for i in range(self.timesteps)], 'num_positive': sum(sum(example['mask'][(self.timesteps - 1) * (task_id)][0].float()))})

            elif self.dense:
                #ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss': loc_loss, 'loc_loss_elem': box_loss.detach().cpu(), 'num_positive': sum(sum(example['mask'][task_id // 2][task_id % 2].float()))})
                ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss': loc_loss, 'loc_loss_elem': box_loss.detach().cpu(), 'num_positive': sum(sum(example['mask'][task_id][0].float()))})

            else:
                ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss': loc_loss, 'loc_loss_elem': [box_loss[i].detach().cpu() for i in range(self.timesteps)], 'num_positive': sum(sum(sum([example['mask'][i][task_id].float() for i in range(self.timesteps)])))})
            
            rets.append(ret)

        """convert batch-key to key-batch
        """

        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing 
        """
        # get loss info
        rets = []
        metas = []

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        forecast_preds_dicts = []

        if self.standard or self.reverse:
            preds_dict = preds_dicts[0]
            vels = [preds_dict['vel'][:,2*i:2*i+2] for i in range(self.timesteps)]
            
            if len(vels) == 1:
                vels = self.target_timesteps * vels

            self.num_classes = [1] * self.target_timesteps

            for vel in vels:
                preds_dict["vel"] = vel
                forecast_preds_dicts.append(copy.deepcopy(preds_dict))

        elif self.sparse:
            forward_dict = preds_dicts[0]
            reverse_dict = preds_dicts[1]

            forward_vels = [forward_dict['vel'][:,2*i:2*i+2] for i in range(self.timesteps)]
            reverse_vels = [reverse_dict['vel'][:,2*i:2*i+2] for i in range(self.timesteps)]

            self.num_classes = [1, 1] * self.target_timesteps

            for vel in forward_vels:
                forward_dict["vel"] = vel
                forecast_preds_dicts.append(copy.deepcopy(forward_dict))
            
            for vel in reverse_vels:
                reverse_dict["vel"] = vel
                forecast_preds_dicts.append(copy.deepcopy(reverse_dict))

        else:
            forecast_preds_dicts = preds_dicts

        for task_id, preds_dict in enumerate(forecast_preds_dicts):
            # convert N C H W to N H W C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
        
            batch_hm = torch.sigmoid(preds_dict['hm'])

            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)

            batch_rot = batch_rot.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            if 'rvel' in preds_dict:
                batch_vel = preds_dict['vel']
                batch_vel = batch_vel.reshape(batch, H*W, 2)
                
                batch_rvel = preds_dict['rvel']
                batch_rvel = batch_rvel.reshape(batch, H*W, 2)
                
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rvel, batch_rot], dim=2)
            
            elif 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
                batch_vel = batch_vel.reshape(batch, H*W, 2)
                
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
            else: 
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            metas.append(meta_list)

            if test_cfg.get('per_class_nms', False):
                pass 
            else:
                rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range, task_id)) 

        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list 


    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range, task_id):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask 

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]

            if batch_box_preds.shape[-1] == 9:
                boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            else:
                boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -2]]

            if test_cfg.get('circular_nms', False):
                centers = boxes_for_nms[:, [0, 1]] 
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = _circle_nms(boxes, min_radius=test_cfg.min_radius[task_id], post_max_size=test_cfg.nms.nms_post_max_size)  
            else:
                selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms.float(), scores.float(), 
                                    thresh=test_cfg.nms.nms_iou_threshold,
                                    pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                    post_max_size=test_cfg.nms.nms_post_max_size)

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts 

import numpy as np 
def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep  