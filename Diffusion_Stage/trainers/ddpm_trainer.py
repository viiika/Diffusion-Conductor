import torch
import torch.nn.functional as F
import random
import time
from models.transformer import MotionTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as torch_f
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist

import numpy as np

from mmcv.runner import get_dist_info
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

from datasets import build_dataloader


class DDPMTrainer(object):

    def __init__(self, args, encoder):
        self.opt = args
        
        self.device = args.device
        
        # print('ddpmtraniner device',self.device)
        self.encoder = encoder
        self.diffusion_steps = args.diffusion_steps
        sampler = 'uniform'
        beta_scheduler = 'linear'
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            # model_mean_type=ModelMeanType.EPSILON, # change here
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        self.sampler_name = sampler

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data, eval_mode=False):
        
        caption, motions, m_lens = batch_data
        #print("caption.shape, motions.shape ", caption.shape, motions.shape) # torch.Size([32, 5400, 128]) torch.Size([32, 1800, 13, 2])
        motions = motions.detach().to(self.device).float()

        self.caption = caption # torch.Size([1568, 540, 128])
        self.motions = motions # torch.Size([1568, 180, 13, 2])
        
        # print('self.caption: ', self.caption.shape) 
        # print('self.motions: ', self.motions.shape)
        
        x_start = motions
        B, T = x_start.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
        t, _ = self.sampler.sample(B, x_start.device)
        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={"text": caption, "length": cur_len}
        )

        self.real_noise = output['target']
        self.fake_noise = output['pred']
        
        self.velocity_body = output['velocity_body']
        self.velocity_elbow = output['velocity_elbow']
        
        try:
            self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)
            
        # testing: generate motion when training (to be continued)
        # batch_size = caption.size(dim=0)
        # print('len: ', batch_size)
        
        # pred_motions = []
        
        # for i in range(batch_size):
        #     cur_mel = caption[i].unsqueeze(0).to(self.device)
        #     print('cur_mel: ', cur_mel.shape)
        
        #     xf_proj, xf_out = self.encoder.module.encode_music(cur_mel, self.device)
        #     B = cur_mel.shape[0]
        #     T = xf_proj.shape[1]#1800#music_mel.shape[1]=5400?
            
        #     dim_pose = 26
            
        #     output = self.diffusion.ddim_sample_loop(
        #         self.encoder,
        #         (B, T, dim_pose),
        #         clip_denoised=False,
        #         progress=True,
        #         model_kwargs={
        #             'xf_proj': xf_proj,
        #             'xf_out': xf_out,
        #             'length': torch.LongTensor([T] * B).to(self.device)
        #         })
            
        #     pred_motions.append(output.reshape(output.shape[0], output.shape[1], 13, 2))
            
        #     print('pred_motions: ', len(pred_motions))
        
        # cur_mel = caption.to(self.device)
        # # print('cur_mel: ', cur_mel.shape)
    
        # xf_proj, xf_out = self.encoder.module.encode_music(cur_mel, self.device)
        # B = cur_mel.shape[0]
        # T = xf_proj.shape[1]#1800#music_mel.shape[1]=5400?
        
        # dim_pose = 26
        
        # output = self.diffusion.ddim_sample_loop(
        #     self.encoder,
        #     (B, T, dim_pose),
        #     noise=self.fake_noise.detach(),
        #     clip_denoised=False,
        #     progress=True, # True to False
        #     model_kwargs={
        #         'xf_proj': xf_proj,
        #         'xf_out': xf_out,
        #         'length': torch.LongTensor([T] * B).to(self.device)
        #     })
        
        # self.pred_motions = output.reshape(output.shape[0], output.shape[1], 13, 2)

    def generate_batch(self, caption, m_lens, dim_pose, idxs=[]):
        xf_proj, xf_out = self.encoder.encode_text(caption, self.device)
        
        B = len(caption)
        T = min(m_lens.max(), self.encoder.num_frames)
        output = self.diffusion.p_sample_loop(
            self.encoder,
            (B, T, dim_pose),
            clip_denoised=False,
            progress=True, # True to False
            model_kwargs={
                'xf_proj': xf_proj,
                'xf_out': xf_out,
                'length': m_lens
            },
            idxs=idxs)
        return output

    def generate_music_motion(self, music_mel, dim_pose, batch_size=1024, idxs=[]):
        # print(music_mel.shape)#(44735, 128)
        music_mel = torch.from_numpy(music_mel).unsqueeze(0).to(self.device)
       
        xf_proj, xf_out = self.encoder.encode_music(music_mel, self.device)
        B = music_mel.shape[0]
        T = xf_proj.shape[1]#1800#music_mel.shape[1]=5400?
        # print('after encode music')
        # print(B,T,xf_proj.shape)
        output = self.diffusion.p_sample_loop(
            self.encoder,
            (B, T, dim_pose),
            clip_denoised=False,
            progress=True, # True to False
            model_kwargs={
                'xf_proj': xf_proj,
                'xf_out': xf_out,
                'length': torch.LongTensor([T] * B).to(self.device)
            },
            idxs=idxs)
        return output


    def generate(self, caption, m_lens, dim_pose, batch_size=1024):
        N = len(caption)
        cur_idx = 0
        self.encoder.eval()
        all_output = []
        while cur_idx < N:
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            output = self.generate_batch(batch_caption, batch_m_lens, dim_pose)
            B = output.shape[0]

            for i in range(B):
                all_output.append(output[i])
            cur_idx += batch_size
        return all_output

    def backward_G(self):
        # motion loss
        # print('self.pred_motions shape: ', self.pred_motions.shape)
        # print('self.motions shape: ', self.motions.shape)
        
        # print('self.pred_motions: ', self.pred_motions)
        # print('self.motions: ', self.motions)
        
        # print('np mse: ', np.mean((self.pred_motions.detach().cpu().numpy() - self.motions.detach().cpu().numpy()) ** 2))
        
        # loss_mot_xy = torch_f.mse_loss(self.pred_motions, self.motions) # motion reconstruction loss
        # self.loss_mot_xy = loss_mot_xy
        
        loss_mot_rec = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
        loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
        self.loss_mot_rec = loss_mot_rec
        
        elbow_lambda = 0.1
        
        # loss_velocity = - elbow_lambda * self.velocity_elbow + self.velocity_body
        loss_velocity = self.velocity_body
        loss_velocity = torch.clamp(loss_velocity, min = -0.1, max = 0.1)
        self.loss_velocity = loss_velocity
        
        # self.loss = 0.1 * loss_mot_xy + loss_mot_rec
        # self.loss = loss_mot_rec + loss_velocity
        self.loss = loss_mot_rec
        
        loss_logs = OrderedDict({})
        
        # loss_logs['loss_mot_xy'] = 0.1 * self.loss_mot_xy.item()
        loss_logs['loss_mot_rec'] = self.loss_mot_rec.item()
        loss_logs['loss_velocity'] = self.loss_velocity.item()
        
        # print('loss_logs[loss_mot_xy]: ', loss_logs['loss_mot_xy'])
        # print('loss_logs[loss_mot_rec]: ', loss_logs['loss_mot_rec'])
        
        return loss_logs

    def update(self):
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()
        self.loss.backward()
        # self.loss_mot_rec.backward()
        self.clip_norm([self.encoder])
        self.step([self.opt_encoder])

        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.encoder = self.encoder.to(device)

    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval()

    def save(self, file_name, ep, total_it):
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except:
            state['encoder'] = self.encoder.state_dict()
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        
        # using data parallel needs to replace 'module.'
        new_checkpoint = checkpoint
        new_encoder = {}
        
        checkpoint_encoder = checkpoint['encoder']

        if self.opt.is_train:
            for key in list(checkpoint_encoder.keys()):
                new_encoder['module.' + key] = checkpoint_encoder[key]
                new_checkpoint['encoder'] = new_encoder

        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        self.encoder.load_state_dict(new_checkpoint['encoder'], strict=False) # True
        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.opt.lr)
        it = 0
        cur_epoch = 0

        if self.opt.is_continue:
            # print(pjoin(self.opt.model_dir, 'latest.tar'))
            print(pjoin(self.opt.model_dir, 'latest.tar'))
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it = self.load(model_dir)

        start_time = time.time()

        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            workers_per_gpu=4,
            shuffle=True,
            dist=self.opt.distributed,
            num_gpus=len(self.opt.gpu_id))

        logs = OrderedDict()
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            for i, batch_data in enumerate(train_loader):
                self.forward(batch_data)
                log_dict = self.update()
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                it += 1
                if it % self.opt.log_every == 0 and rank == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)

                if it % self.opt.save_latest == 0 and rank == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if rank == 0:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if epoch % self.opt.save_every_e == 0 and rank == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar'%(epoch)),
                            epoch, total_it=it)
