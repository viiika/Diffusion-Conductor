import os
from os.path import join as pjoin

import torch
import torch.distributed as dist
import torch.nn as nn

from mmcv.runner import get_dist_info

from scipy import linalg
import scipy.signal as scisignal

import numpy as np
import librosa
import argparse

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
        
        base_weights = torch.load('/home/zhuoran/code/Diffusion_Stage/stage_one_checkpoints/M2SNet_latest.pt')

        new_weights = {}
        for key in list(base_weights.keys()):
            if key.startswith('module.motion_encoder'):
                new_weights[key.replace('module.motion_encoder.', '')] = base_weights[key]
                
        self.motion_encoder.load_state_dict(new_weights, strict=True)
        self.motion_encoder.eval()

# Evaluator (FGD, BC, Diversity)
class Evaluator():
    def __init__(self):
        motion_pretrain = MotionPretrain()
        self.motion_encoder = motion_pretrain.motion_encoder
        
        # motion list
        self.real_motion_list = []
        self.generated_motion_list = []
        
        # motion latent feature list
        self.real_motion_latent_list = []
        self.generated_motion_latent_list = []
        
        # real motion beat scores and generated motion beat scores
        self.real_beat_scores = []
        self.generated_beat_scores = []
        
        self.num_samples = None

    def push_samples(self, opt, dim_pose, device):
        encoder = build_models(opt, dim_pose).to(device)
    
        opt.dim_pose = 26

        trainer = DDPMTrainer(opt, encoder)
        trainer.load(pjoin(opt.model_dir, 'latest.tar'))
        
        trainer.eval_mode()
        trainer.to(opt.device)
        
        path = '/mnt/data/zhuoran/test'
        folder = os.listdir(path)
        num = len(folder)
        self.num_samples = num
        
        with torch.no_grad():
            # num -> 10
            for i in range(num):
                id = folder[i]
                cur_path = path + '/' + str(id)
                
                mel = np.load(cur_path + '/mel.npy') # (5400, 128)
                motion = np.load(cur_path + '/motion.npy') # (1800, 13, 2)
                
                pred_motions = trainer.generate_music_motion(mel, opt.dim_pose)
                pred_motion = pred_motions[0].cpu().numpy()
                pred_motion = pred_motion.reshape([pred_motion.shape[0],13,2])
                
                # convert to latent features
                real_feat = self.motion_encoder.features(torch.from_numpy(motion).unsqueeze(0))[-1] # (1, 1800, 64)
                generated_feat = self.motion_encoder.features(torch.from_numpy(pred_motion).unsqueeze(0))[-1] # (1, 1800, 64)
                
                self.real_motion_list.append(motion)
                self.generated_motion_list.append(pred_motion)
                
                self.real_motion_latent_list.append(real_feat[0]) # push (1800, 64) feat into latent_list
                self.generated_motion_latent_list.append(generated_feat[0]) # push (1800, 64) feat into latent_list
                
                real_beat_score = self.calculate_beat_scores(motion, mel, 'real')
                generated_beat_score = self.calculate_beat_scores(pred_motion, mel, 'generated')
                
                self.real_beat_scores.append(real_beat_score)
                self.generated_beat_scores.append(generated_beat_score)
    
    # Calculate diversity scores
    def get_diversity_scores(self):
        feat1 = np.vstack(self.generated_motion_latent_list[:500])
        random_idx = torch.randperm(len(self.generated_motion_latent_list))[:500]
        shuffle_list = [self.generated_motion_latent_list[x] for x in random_idx]
        feat2 = np.vstack(shuffle_list)
        
        feat_dist = np.mean(np.sum(np.absolute(feat1 - feat2), axis=-1))
        return feat_dist

    # Calculate frechet distance and latent feature distance
    def get_scores(self):
        generated_feats = np.vstack(self.generated_motion_latent_list)
        real_feats = np.vstack(self.real_motion_latent_list)

        def frechet_distance(samples_A, samples_B):
            A_mu = np.mean(samples_A, axis=0)
            A_sigma = np.cov(samples_A, rowvar=False)
            B_mu = np.mean(samples_B, axis=0)
            B_sigma = np.cov(samples_B, rowvar=False)
            try:
                frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
            except ValueError:
                frechet_dist = 1e+10
            return frechet_dist

        ####################################################################
        # frechet distance
        frechet_dist = frechet_distance(generated_feats, real_feats)

        ####################################################################
        # distance between real and generated samples on the latent feature space
        dists = []
        for i in range(real_feats.shape[0]):
            d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
            dists.append(d)
        feat_dist = np.mean(dists)

        return frechet_dist, feat_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
    
    def alignment_score(self, music_beats, motion_beats, motion_type, sigma=3):
        """Calculate alignment score between music and motion."""
        if motion_beats.sum() == 0:
            return 0.0
        music_beat_idxs = np.where(music_beats)[0] # (124,)
        motion_beat_idxs = np.where(motion_beats)[0] # (80,)

        score_all = []
        
        # Beat Align (Proposed in: AI Choreographer: Music Conditioned 3D Dance Generation with AIST++)
        # for motion_beat_idx in motion_beat_idxs:
        #     dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        #     ind = np.argmin(dists)            
        #     score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        #     score_all.append(score)
        
        # Beat Consistency (Proposed in: DanceFormer: Music Conditioned 3D Dance Generation with Parametric Motion Transformer)
        for music_beat_idx in music_beat_idxs:
            dists = np.abs(music_beat_idx - motion_beat_idxs).astype(np.float32)
            ind = np.argmin(dists)            
            score = np.exp(- dists[ind]**2 / 2 / sigma**2)
            score_all.append(score)
        return sum(score_all) / len(score_all)
    
    def normalize(self, arr):
        """
        Normalize a one-dimensional array.
        """
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    def motion_peak_onehot(self, joints, motion_type):
        """Calculate motion beats.
        Kwargs:
            joints: [nframes, njoints, 3]
        Returns:
            - peak_onhot: motion beats.
        """
        # Calculate velocity.
        velocity = np.zeros_like(joints, dtype=np.float32) # joint shape: (1800, 13, 2)
        velocity[1:] = joints[1:] - joints[:-1]
        velocity_norms = np.linalg.norm(velocity, axis=2)
        
        envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)
        norm_envelope = self.normalize(envelope)

        # Find local minima in velocity -- beats
        peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10) # 10 for 60FPS
        peak_onehot = np.zeros_like(envelope, dtype=bool)
        peak_onehot[peak_idxs] = 1

        # # Second-derivative of the velocity shows the energy of the beats
        # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
        # # optimize peaks
        # peak_onehot[peak_energy<0.001] = 0
        return peak_onehot
    
    def calculate_beat_scores(self, cur_motion, cur_music, motion_type):
        motion_beats = self.motion_peak_onehot(cur_motion, motion_type) # (5400,)
        audio_beat_time = self.get_music_beat(cur_music) # (1800,)

        beat_score = self.alignment_score(audio_beat_time, motion_beats, motion_type, sigma=3)
        
        return beat_score

    def get_music_beat(self, data):
        FPS = 90 # 60
        HOP_LENGTH = 512
        SR = FPS * HOP_LENGTH
        EPS = 1e-6
        
        # 参数说明
        # S: pre-computed (log-power) spectrogram
        # Use transpose to make data correspond to [shape=(..., d, m)]
        envelope = librosa.onset.onset_strength(S=np.transpose(data), sr=SR)

        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH) # sr 
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0  # (seq_len,)

        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH, tightness=100)
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0  # (seq_len,)
        
        return beat_onehot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, default="/home/zhuoran/code/Diffusion_Stage/checkpoints/ConductorMotion100/add_velocity_pretrain_elbow_clamp_lambda/opt.txt", help='Opt path')
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
        opt.data_root = '/mnt/data/zhuoran/'
        opt.joints_num = 13
        dim_pose = 26 #[1800, 13, 2]
        opt.max_motion_length = 1800
        sample_length = 30 # means 60s
        
        split = 'test'
        limit = None
        root_dir = '/mnt/data/zhuoran/'
    else:
        raise KeyError('Dataset Does Not Exist')
    
    path = '/mnt/data/zhuoran/test'

    evaluator = Evaluator()
    evaluator.push_samples(opt, dim_pose, device)

    diversity_score = evaluator.get_diversity_scores()
    frechet_dist, feat_dist = evaluator.get_scores()
    
    print('FGD: ', frechet_dist)
    print('diversity_score: ', diversity_score)
    print('feat_dist: ', feat_dist)
    
    print ("\nBeat score on real data: %.3f\n" % (sum(evaluator.real_beat_scores) / len(evaluator.real_beat_scores)))
    print ("\nBeat score on generated data: %.3f\n" % (sum(evaluator.generated_beat_scores) / len(evaluator.generated_beat_scores)))
