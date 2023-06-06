import os
from os.path import join as pjoin

import torch
import torch.distributed as dist
import torch.nn as nn

from mmcv.runner import get_dist_info, init_dist

import numpy as np

import argparse

import utils.paramUtil as paramUtil
from utils.plot_script import *
from models import MotionTransformer
from trainers import DDPMTrainer
from utils.get_opt import get_opt
from models.ST_GCN.ST_GCN import ST_GCN

torch.cuda.set_device(1)
os.environ['CUDA_VISIBLE_DEVICES'] ='1, 2'

def build_models(opt, dim_pose):
    encoder = MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        device = opt.device,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder

class MotionEncoder_STGCN(nn.Module):
    def __init__(self):
        super(MotionEncoder_STGCN, self).__init__()
        self.graph_args = {}
        self.st_gcn = ST_GCN(in_channels=2,
                             out_channels=32,
                             graph_args=self.graph_args,
                             edge_importance_weighting=True,
                             mode='M2S')
        self.fc = nn.Sequential(nn.Conv1d(32 * 13, 64, kernel_size=1), nn.BatchNorm1d(64)) # change 64 into 512

    def forward(self, input):
        input = input.transpose(1, 2)
        input = input.transpose(1, 3)
        input = input.unsqueeze(4)

        output = self.st_gcn(input)
        output = output.transpose(1, 2)
        output = torch.flatten(output, start_dim=2)
        output = self.fc(output.transpose(1, 2)).transpose(1, 2)

        return output

    def features(self, input):
        input = input.transpose(1, 2)
        input = input.transpose(1, 3)
        input = input.unsqueeze(4)

        output = self.st_gcn(input)
        output = output.transpose(1, 2)
        output = torch.flatten(output, start_dim=2)
        output = self.fc(output.transpose(1, 2)).transpose(1, 2)

        features = self.st_gcn.extract_feature(input)
        features.append(output.transpose(1, 2))

        return features
    
class MotionPretrain():
    def __init__(self):
        super(MotionPretrain, self).__init__()
        self.motion_encoder = MotionEncoder_STGCN()
        
        # load pretrain motion encoder
        base_weights = torch.load('/home/zhuoran/DiffuseConductor/Diffusion_Stage/stage_one_checkpoints/M2SNet_latest.pt')

        new_weights = {}
        for key in list(base_weights.keys()):
            if key.startswith('module.motion_encoder'):
                new_weights[key.replace('module.motion_encoder.', '')] = base_weights[key]
                
        self.motion_encoder.load_state_dict(new_weights, strict=True)
        self.motion_encoder.eval()

# Sync Error (SE)
def mse_loss_latent(gt_motion, pred_motion):
    motion_pretrain = MotionPretrain()
    motion_encoder = motion_pretrain.motion_encoder
    
    gt_features = motion_encoder.features(torch.from_numpy(gt_motion).unsqueeze(0))[-1]
    pred_features = motion_encoder.features(torch.from_numpy(pred_motion).unsqueeze(0))[-1]
    
    sq_diff = (pred_features - gt_features) ** 2
    mse_loss_latent = np.mean(sq_diff.numpy())
    
    return mse_loss_latent

# Mean Square Error (MSE)
def mse_loss(gt_motion, pred_motion):
    sq_diff = (pred_motion - gt_motion) ** 2

    # Calculate the MSE loss across all dimensions and joints
    mse_loss = np.mean(sq_diff)
    
    return mse_loss

# Evaluator (MSE, SE)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, default="/home/zhuoran/DiffuseConductor/Diffusion_Stage/checkpoints/ConductorMotion100/add_velocity_pretrain_no_velocity/opt.txt", help='Opt path')
    parser.add_argument('--gpu_id', type=int, default=5, help="which gpu to use")
    
    args = parser.parse_args()
    device = torch.device('cuda:%d' % args.gpu_id if args.gpu_id != -1 else 'cpu')
    opt = get_opt(args.opt_path, device)
    rank, world_size = get_dist_info()
    opt.do_denoise = True
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    if opt.dataset_name == 'ConductorMotion100':
        opt.data_root = '/mnt/data/zhuoran/' # dataset root path
        
        opt.joints_num = 13
        dim_pose = 26 #[1800, 13, 2]
        opt.max_motion_length = 1800
        sample_length = 30 # means 60s
        
        split = 'test'
        limit = None
        root_dir = '/mnt/data/zhuoran/' # dataset root path

    elif opt.dataset_name == 'kit':
        opt.data_root = './data/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain

    else:
        raise KeyError('Dataset Does Not Exist')

    encoder = build_models(opt, dim_pose).to(device)
    opt.dim_pose = 26 # 13*2

    trainer = DDPMTrainer(opt, encoder)
    trainer.load(pjoin(opt.model_dir, 'latest.tar'))

    trainer.eval_mode()
    trainer.to(opt.device)
    
    path = '/mnt/data/zhuoran/test' # test set path
    folder = os.listdir(path)
    
    total_loss = 0
    total_latent_loss = 0
    
    num = len(folder)
    
    with torch.no_grad():
      for i in range(num):
        id = folder[i]
        cur_path = path + '/' + str(id)
        
        mel = np.load(cur_path + '/mel.npy')
        motion = np.load(cur_path + '/motion.npy')
        
        pred_motions = trainer.generate_music_motion(mel, opt.dim_pose)
        pred_motion = pred_motions[0].cpu().numpy()
        pred_motion = pred_motion.reshape([pred_motion.shape[0],13,2])
        
        cur_latent_loss = mse_loss_latent(motion, pred_motion)
        total_latent_loss += cur_latent_loss
        
        cur_loss = mse_loss(motion, pred_motion)
        total_loss += cur_loss
    
    final_mse = total_loss / num
    print('final_mse: ', final_mse) 
    
    final_latent_mse = total_latent_loss / num
    print('final_latent_mse: ', final_latent_mse)
