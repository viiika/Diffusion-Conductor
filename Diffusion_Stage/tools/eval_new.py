import os
from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *

from models import MotionTransformer
from trainers import DDPMTrainer
from datasets import Text2MotionDataset, Music2MotionDataset

from mmcv.runner import get_dist_info, init_dist
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
import torch
import torch.distributed as dist

import numpy as np

import argparse

from utils.get_opt import get_opt

# torch.cuda.set_device(6)
# os.environ['CUDA_VISIBLE_DEVICES'] ='6, 7'

torch.cuda.set_device(1)
os.environ['CUDA_VISIBLE_DEVICES'] ='1, 2, 3'

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

def mse_loss(gt_motion, pred_motion):
    # Calculate the squared difference between the generated and ground-truth motion
    sq_diff = (pred_motion - gt_motion) ** 2

    # Calculate the MSE loss across all dimensions and joints
    mse_loss = np.mean(sq_diff)
    
    return mse_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, default="/Users/jinbin/5340Proj/alice_dir/my_MotionDiffuse/ssh_MotionDiffuse/text2motion/checkpoints/ConductorMotion100/alice_version_9/opt.txt", help='Opt path')
    parser.add_argument('--gpu_id', type=int, default=5, help="which gpu to use")
  
    # parser = TrainCompOptions()
    # opt = parser.parse()
    
    args = parser.parse_args()
    
    device = torch.device('cuda:%d' % args.gpu_id if args.gpu_id != -1 else 'cpu')

    opt = get_opt(args.opt_path, device)
    
    rank, world_size = get_dist_info()

    # opt.device = torch.device("cuda")
    # torch.autograd.set_detect_anomaly(True)

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
        opt.data_root = '/Users/jinbin/5340Proj/dataset/'
        # opt.motion_dir = ''#pjoin(opt.data_root, 'new_joint_vecs')
        # opt.text_dir = ''#pjoin(opt.data_root, 'texts')
        opt.joints_num = 13
        dim_pose = 26 #[1800, 13, 2]
        # radius = 4
        #dim_mel [5400, 128]
        opt.max_motion_length = 1800
        # sample_length = 60 # means 60s
        # sample_length = 6 # means 60s
        sample_length = 30 # means 60s
        
        split = 'test'
        limit = None
        root_dir = '/Users/jinbin/5340Proj/dataset/'

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
    # if world_size > 1:
    #     encoder = MMDistributedDataParallel(
    #         encoder.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False,
    #         find_unused_parameters=True)
    # elif opt.data_parallel:
    #     encoder = MMDataParallel(
    #         encoder.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
    # else:
    #     encoder = encoder.cuda()
    
    opt.dim_pose = 26

    trainer = DDPMTrainer(opt, encoder)
    trainer.load(pjoin(opt.model_dir, 'latest.tar'))  
    
    trainer.eval_mode()
    trainer.to(opt.device)
    
    path = '/Users/jinbin/5340Proj/dataset/test'
    folder = os.listdir(path)
    
    total_loss = 0
    
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
        
        cur_loss = mse_loss(motion, pred_motion)
        
        total_loss += cur_loss
        
        print('cur_loss: ', cur_loss)
        print('total_loss: ', total_loss)
    
    print('final total loss: ', total_loss)
    final_mse = total_loss / num
    print('final_mse: ', final_mse)   
