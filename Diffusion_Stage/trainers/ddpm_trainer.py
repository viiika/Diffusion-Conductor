import torch
import torch.nn.functional as F
import time
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as torch_f
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin

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
from models.ST_GCN.ST_GCN import ST_GCN


# Motion Encoder
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

# Load pretrain motion encoder
class MotionPretrain():
    def __init__(self):
        super(MotionPretrain, self).__init__()
        self.motion_encoder = MotionEncoder_STGCN()
        
        base_weights = torch.load('/home/zhuoran/DiffuseConductor/Diffusion_Stage/stage_one_checkpoints/M2SNet_latest.pt')

        new_weights = {}
        for key in list(base_weights.keys()):
            if key.startswith('module.motion_encoder'):
                new_weights[key.replace('module.motion_encoder.', '')] = base_weights[key]
                
        self.motion_encoder.load_state_dict(new_weights, strict=True)
        self.motion_encoder.eval()


class DDPMTrainer(object):
    def __init__(self, args, encoder):
        self.opt = args
        self.device = args.device
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
        
        motion_pretrain = MotionPretrain()
        self.motion_encoder = motion_pretrain.motion_encoder.to(self.device)
        
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
        motions = motions.detach().to(self.device).float()

        self.caption = caption # torch.Size([1568, 540, 128])
        self.motions = motions # torch.Size([1568, 180, 13, 2])
        
        x_start = motions
        B, T = x_start.shape[:2] # batch_size, 900
        
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
        self.velocity = output['velocity']
        
        try:
            self.src_mask = self.encoder.module.generate_src_mask(64, cur_len).to(x_start.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(64, cur_len).to(x_start.device)

    def generate_batch(self, caption, m_lens, dim_pose, idxs=[]):
        xf_proj, xf_out = self.encoder.encode_text(caption, self.device)
        
        B = len(caption)
        T = min(m_lens.max(), self.encoder.num_frames)
        output = self.diffusion.ddim_sample_loop(
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
        music_mel = torch.from_numpy(music_mel).unsqueeze(0).to(self.device)
       
        xf_proj, xf_out = self.encoder.encode_music(music_mel, self.device)
        B = music_mel.shape[0]
        T = xf_proj.shape[1]

        output = self.diffusion.ddim_sample_loop(
            self.encoder,
            (B, T, dim_pose),
            clip_denoised=False,
            progress=True,
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
        # Diffusion loss
        fake_noise = self.fake_noise.reshape([self.fake_noise.shape[0], self.fake_noise.shape[1], int(self.fake_noise.shape[2]/2), 2]).to(self.device)
        real_noise = self.real_noise.reshape([self.real_noise.shape[0], self.real_noise.shape[1], int(self.real_noise.shape[2]/2), 2]).to(self.device)
        
        fake_noise_feat = self.motion_encoder.features(fake_noise)[-1] # (40, 64, 900)
        real_noise_feat = self.motion_encoder.features(real_noise)[-1]
        
        # loss_mot_rec = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1) # (64, 900, 26)
        loss_mot_rec = self.mse_criterion(fake_noise_feat, real_noise_feat).mean(dim=-1) # (64, 900, 26)
        
        loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
        self.loss_mot_rec = loss_mot_rec

        # Velocity loss
        loss_velocity = self.velocity
        self.loss_velocity = loss_velocity
        
        # Elbow loss
        loss_velocity_elbow = torch.clamp(self.velocity_elbow, min = -0.0002, max = 0.0002)
        self.loss_velocity_elbow = loss_velocity_elbow
        
        lambda_velocity = 0.1
        lambda_elbow = 0.1

        self.loss = loss_mot_rec + lambda_velocity * loss_velocity - lambda_elbow * loss_velocity_elbow
        
        loss_logs = OrderedDict({})
        
        loss_logs['loss_mot_rec'] = self.loss_mot_rec.item()
        loss_logs['loss_velocity'] = self.loss_velocity.item()
        loss_logs['loss_elbow'] = self.loss_velocity_elbow.item()
        
        return loss_logs

    def update(self):
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()
        self.loss.backward()
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
        
        new_checkpoint = checkpoint
        new_encoder = {}
        
        checkpoint_encoder = checkpoint['encoder']

        if self.opt.is_train:
            for key in list(checkpoint_encoder.keys()):
                new_encoder['module.' + key] = checkpoint_encoder[key]
                new_checkpoint['encoder'] = new_encoder

        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        self.encoder.load_state_dict(new_checkpoint['encoder'], strict=False)
        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.opt.lr)
        it = 0
        cur_epoch = 0

        if self.opt.is_continue:
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
