import argparse
import time
import os
import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models.M2SNet
from models.Discriminator import Discriminator_1DCNN
from models.Generator import Generator
from utils.dataset import ConductorMotionDataset
from utils.loss import SyncLoss, rhythm_density_error, strengh_contour_error, Motion_features
from utils.train_utils import plot_motion
from scipy import linalg
import scipy.signal as scisignal
import librosa


if torch.cuda.is_available():
    device = "cuda"
    device = torch.device("cuda:7")
else:
    device = "cpu"

class M2SGAN_Evaluator():
    def __init__(self, args):
        self.batch_size = 1
        self.sample_length = args.sample_length
        self.mode = args.mode
        self.save_path = 'checkpoints/M2SGAN/' + self.mode + time.strftime("_%a-%b-%d_%H-%M-%S", time.localtime())
        os.mkdir(self.save_path)

        self.testing_set = ConductorMotionDataset(sample_length=args.sample_length,
                                                  split=args.testing_set,
                                                  limit=args.testing_set_limit,
                                                  root_dir=args.dataset_dir)
        self.test_loader = DataLoader(dataset=self.testing_set, batch_size=self.batch_size, shuffle=False) # shuffle True -> False
        print('testing set initialized, {} samples, {} hours'
              .format(len(self.testing_set), round(len(self.testing_set) * args.sample_length / 3600, 2)))

        self.MSE = nn.MSELoss()

        M2SNet = models.M2SNet.M2SNet().to(device)
        # M2SNet.load_state_dict(torch.load(args.M2SNet))
        paras = torch.load(args.M2SNet)
        new_paras = {}
        for key in paras.keys():
            new_paras[key.replace('module.', '')] = paras[key]
        M2SNet.load_state_dict(new_paras)
        print('Load model from {}'.format(args.M2SNet))
        
        M2SNet.eval()
        self.perceptual_loss = SyncLoss(M2SNet.motion_encoder)
        
        self.real_motion_latent_list = []
        self.generated_motion_latent_list = []

    def evaluate(self, G, D, perceptual_loss, writer, epoch, total_step, motion_features, save_checkpoints=True):
        G.eval()
        D.eval()

        print('| Evaluating M2SGAN at Epoch {}'.format(epoch))

        # Realism
        SD_fake_all = []
        SD_real_all = []
        W_dis_all = []

        # Consistency
        MSE_all = []
        MPE_all = []
        loss_sync_all = []
        RDE_all = []
        SCE_all = []
        
        frechet_distance_all = []
        feat_dist_all = []
        
        real_beat_scores = []
        generated_beat_scores = []

        pbar = tqdm.tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        for step, (mel, real_motion) in pbar:
            if real_motion.shape[0] != self.batch_size:
                continue
            mel = mel.type(torch.FloatTensor).to(device) # (1, 2700, 128)
            real_motion = real_motion.type(torch.FloatTensor).to(device) # (1, 900, 13, 2)

            noise = torch.randn([self.batch_size, self.sample_length, 8]).to(device)
            fake_motion = G(mel, noise)

            # ----------- #
            #   Realism   #
            # ----------- #

            # Standard Deviation
            fake_sd = torch.mean(torch.std(fake_motion, dim=1)).item()
            real_sd = torch.mean(torch.std(real_motion, dim=1)).item()
            SD_fake_all.append(fake_sd)
            SD_real_all.append(real_sd)

            # W Distance
            real_output_D = D(real_motion)
            fake_output_D = D(fake_motion.detach())
            W_dis_all.append((real_output_D - fake_output_D).detach().cpu().numpy().mean())

            # ----------- #
            # Consistency #
            # ----------- #

            # Mean Squared Error
            mse = self.MSE(fake_motion, real_motion)
            MSE_all.append(mse.item())

            # Mean Perceptual Error
            mpe = self.perceptual_loss(fake_motion, real_motion)
            MPE_all.append(mpe.item())

            # Perceptual Loss
            loss_sync = perceptual_loss(fake_motion, real_motion)
            loss_sync_all.append(loss_sync.item())

            # Rhythm Density Error
            RDE = rhythm_density_error(real_motion, fake_motion)
            RDE_all.append(RDE)

            # Strengh Contur Error
            SCE = strengh_contour_error(real_motion, fake_motion)
            SCE_all.append(SCE.item())
            
            # Latent Motion Feature
            real_feats, generated_feats = self.get_latent_motion(real_motion, fake_motion)
            
            # Frechet Distance
            frechet_dist, feat_dist = self.get_scores(real_feats, generated_feats)
            
            frechet_distance_all.append(frechet_dist)
            feat_dist_all.append(feat_dist)
            
            real_beat_score = self.calculate_beat_scores(real_motion[0].cpu().data.numpy(), mel[0].cpu().data.numpy())
            generated_beat_score = self.calculate_beat_scores(fake_motion[0].cpu().data.numpy(), mel[0].cpu().data.numpy())
            
            real_beat_scores.append(real_beat_score)
            generated_beat_scores.append(generated_beat_score)

        writer.add_scalars('M2SGAN_Realism/W_distance',
                           {'test': np.array(W_dis_all).mean()}, total_step)
        writer.add_scalars('M2SGAN_Realism/Standard Deviation',
                           {'test': np.mean(SD_fake_all),
                            'real': np.mean(SD_real_all)}, total_step)

        writer.add_scalars('M2SGAN_Consistency/MSE Loss',
                           {'test': np.mean(MSE_all)}, total_step)
        writer.add_scalars('M2SGAN_Consistency/Sync Loss',
                           {'test': np.mean(loss_sync_all)}, total_step)

        writer.add_scalars('M2SGAN_Consistency/Sync Error (SE)', {'test': np.mean(MPE_all)}, total_step)
        writer.add_scalars('M2SGAN_Consistency/Rhythm Density Error (RDE)', {'test': np.mean(RDE_all)}, total_step)
        writer.add_scalars('M2SGAN_Consistency/Strengh Contour Error (SCE)', {'test': np.mean(SCE_all)}, total_step)

        print('| MPE: %.5f | RDE: %.5f | SCE: %.5f' % (np.mean(MPE_all), np.mean(RDE_all), np.mean(SCE_all)))
        
        print('MSE: ', np.mean(MSE_all))
        
        print('Frechet_distance: %.5f' % (np.mean(frechet_distance_all)))
        
        diversity_score = self.get_diversity_scores()
        print('diversity_score: ', diversity_score)
        
        real_diversity_score = self.get_real_diversity_scores()
        print('real_diversity_score: ', real_diversity_score)
        
        print ("\nBeat score on real data: %.3f\n" % (sum(real_beat_scores) / len(real_beat_scores)))
        print ("\nBeat score on generated data: %.3f\n" % (sum(generated_beat_scores) / len(generated_beat_scores)))

        fig_motion = plot_motion(fake_motion, real_motion)
        writer.add_image("M2SGAN training sample", fig_motion, total_step, dataformats='HWC')

        if save_checkpoints:
            torch.save(G.state_dict(), '{}/M2SGAN_Generator_{}_{}.pt'.format(self.save_path, epoch, total_step))
            torch.save(G.state_dict(), '{}/M2SGAN_Generator_last.pt'.format(self.save_path))
            torch.save(D.state_dict(), '{}/M2SGAN_Discriminator_{}_{}.pt'.format(self.save_path, epoch, total_step))
            torch.save(D.state_dict(), '{}/M2SGAN_Discriminator_last.pt'.format(self.save_path))

        G.train()
        D.train()

        return

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

    def get_scores(self, real_feats, generated_feats):
        # generated_feats = np.vstack(self.generated_motion_latent_list)
        # real_feats = np.vstack(self.real_motion_latent_list)

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
        # real_feats, generated_feats = motion_features(fake_motion, real_motion)
        # real_feats = real_feats[0].cpu().data.numpy()
        # generated_feats = generated_feats[0].cpu().data.numpy()

        frechet_dist = frechet_distance(real_feats, generated_feats)

        ####################################################################
        # distance between real and generated samples on the latent feature space
        dists = []
        for i in range(real_feats.shape[0]):
            d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
            dists.append(d)
        feat_dist = np.mean(dists)

        return frechet_dist, feat_dist

    def get_latent_motion(self, real_motion, fake_motion):
        real_feats, generated_feats = motion_features(fake_motion, real_motion)
        real_feats = real_feats[0].cpu().data.numpy()
        generated_feats = generated_feats[0].cpu().data.numpy()
        
        self.real_motion_latent_list.append(real_feats)
        self.generated_motion_latent_list.append(generated_feats)
        
        return real_feats, generated_feats
    
    def get_diversity_scores(self):
        feat1 = np.vstack(self.generated_motion_latent_list[:500])
        random_idx = torch.randperm(len(self.generated_motion_latent_list))[:500]
        shuffle_list = [self.generated_motion_latent_list[x] for x in random_idx]
        feat2 = np.vstack(shuffle_list)
        
        feat_dist = np.mean(np.sum(np.absolute(feat1 - feat2), axis=-1))
        return feat_dist
    
    def get_real_diversity_scores(self):
        feat1 = np.vstack(self.real_motion_latent_list[:500])
        random_idx = torch.randperm(len(self.real_motion_latent_list))[:500]
        shuffle_list = [self.real_motion_latent_list[x] for x in random_idx]
        feat2 = np.vstack(shuffle_list)
        
        feat_dist = np.mean(np.sum(np.absolute(feat1 - feat2), axis=-1))
        return feat_dist

    def motion_peak_onehot(self, joints):
        """Calculate motion beats.
        Kwargs:
            joints: [nframes, njoints, 3]
        Returns:
            - peak_onhot: motion beats.
        """
        # Calculate velocity.
        velocity = np.zeros_like(joints, dtype=np.float32)
        velocity[1:] = joints[1:] - joints[:-1]
        velocity_norms = np.linalg.norm(velocity, axis=2)
        envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

        # Find local minima in velocity -- beats
        peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10) # 10 for 60FPS
        peak_onehot = np.zeros_like(envelope, dtype=bool)
        peak_onehot[peak_idxs] = 1

        # # Second-derivative of the velocity shows the energy of the beats
        # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
        # # optimize peaks
        # peak_onehot[peak_energy<0.001] = 0
        return peak_onehot

    def calculate_beat_scores(self, cur_motion, cur_music):
        motion_beats = self.motion_peak_onehot(cur_motion) # (2700,)
        audio_beat_time = self.get_music_beat(cur_music) # (900,)
        
        # print('audio_beat_time: ', audio_beat_time)
        # print('motion_beats: ', motion_beats)

        beat_score = self.alignment_score(audio_beat_time, motion_beats, sigma=3)
        
        return beat_score

    def alignment_score(self, music_beats, motion_beats, sigma=3):
        """Calculate alignment score between music and motion."""
        if motion_beats.sum() == 0:
            return 0.0
        music_beat_idxs = np.where(music_beats)[0] # (65,)
        motion_beat_idxs = np.where(motion_beats)[0] # (28,)
        
        # print('music_beat_idxs: ', music_beat_idxs)
        # print('motion_beat_idxs: ', motion_beat_idxs)
        
        score_all = []
        # for motion_beat_idx in motion_beat_idxs:
        #     dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        #     ind = np.argmin(dists)
        #     score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        #     score_all.append(score)
        # Beat Consistency
        for music_beat_idx in music_beat_idxs:
            dists = np.abs(music_beat_idx - motion_beat_idxs).astype(np.float32)
            ind = np.argmin(dists)            
            score = np.exp(- dists[ind]**2 / 2 / sigma**2)
            score_all.append(score)
        return sum(score_all) / len(score_all)
    
    def get_music_beat(self, data):
        FPS = 90 # 60
        HOP_LENGTH = 512
        SR = FPS * HOP_LENGTH
        EPS = 1e-6
        
        # 参数说明
        # S: pre-computed (log-power) spectrogram
        # Use transpose to make data correspond to [shape=(..., d, m)]
        envelope = librosa.onset.onset_strength(S=np.transpose(data), sr=SR)

        # mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
        # chroma = librosa.feature.chroma_cens(
        #     data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T  # (seq_len, 12)

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
    parser = argparse.ArgumentParser(description='Generative Learning Stage')
    parser.add_argument('--M2SNet', default='/home/zhuoran/code/Contrastive_Stage/checkpoints/M2SNet/M2SNet_hard_Latest.pt')#'checkpoints/M2SNet/hard/M2SNet_last.pt')
    parser.add_argument('--transfer_music_encoder', default=True)
    parser.add_argument('--train_music_encoder', default=False)
    parser.add_argument('--M2SNet_test', default='/home/zhuoran/code/Contrastive_Stage/checkpoints/M2SNet/M2SNet_hard_Latest.pt',
                        help='to calculate sync error')

    parser.add_argument('--dataset_dir', default='/mnt/data/zhuoran/')
    parser.add_argument('--training_set', default='train')
    parser.add_argument('--training_set_limit', default=None, help='in: hours')
    parser.add_argument('--testing_set', default='test')
    parser.add_argument('--testing_set_limit', default=None, help='in: hours')

    parser.add_argument('--epoch_num', default=200, help='total epochs')
    parser.add_argument('--evaluate_epoch', default=10, help='interval between performing evaluation')

    parser.add_argument('--batch_size', default=55, type=int, help='batch size')
    parser.add_argument('--sample_length', default=30, help='in: seconds')
    parser.add_argument('--CRITIC_ITERS', default=5)
    parser.add_argument('--lr', default=0.0005, help='learning rate')

    parser.add_argument('--w_adv', default=1, help='weight for adversarial loss')
    parser.add_argument('--w_sync', default=0.05, help='weight for sync loss')
    parser.add_argument('--w_mse', default=0, help='weight for MSE loss')
    parser.add_argument('--w_gp', default=10, help='weight for gradient penalty')

    parser.add_argument('--mode', default='train', help='I donnot know what that is')

    args = parser.parse_args()

    M2SNet = models.M2SNet.M2SNet().to(device)
    # M2SNet.load_state_dict(torch.load('/home/zhuoran/code/Contrastive_Stage/checkpoints/M2SNet/M2SNet_hard_Latest.pt'))
    discriminator_weights = torch.load('/home/zhuoran/code/Contrastive_Stage/checkpoints/M2SGAN/train_Tue-Mar-14_16-43-46/M2SGAN_Discriminator_last.pt')
    # print(discriminator_weights.keys())
    
    generator_weights = torch.load('/home/zhuoran/code/Contrastive_Stage/checkpoints/M2SGAN/train_Tue-Mar-14_16-43-46/M2SGAN_Generator_last.pt')
    # print(generator_weights.keys())
    
    base_weights = torch.load('/home/zhuoran/code/Contrastive_Stage/checkpoints/M2SNet/M2SNet_hard_Latest.pt')
    new_weights = {}
    for key in list(base_weights.keys()):
        if key.startswith('module.'):
            new_weights[key.replace('module.', '')] = base_weights[key]
    
    M2SNet.load_state_dict(new_weights)
    M2SNet.eval()
    perceptual_loss = SyncLoss(M2SNet.motion_encoder)
    motion_features = Motion_features(M2SNet.motion_encoder) # extract latent motion feature
    
    evaluator = M2SGAN_Evaluator(args)
    
    writer = SummaryWriter(comment='_post_eval_')
    
    G = Generator().to(device)
    G.load_state_dict(torch.load('/home/zhuoran/code/Contrastive_Stage/checkpoints/M2SGAN/train_Tue-Mar-14_16-43-46/M2SGAN_Generator_last.pt'))
    D = Discriminator_1DCNN().to(device)
    D.load_state_dict(torch.load('/home/zhuoran/code/Contrastive_Stage/checkpoints/M2SGAN/train_Tue-Mar-14_16-43-46/M2SGAN_Discriminator_last.pt'))

    evaluator.evaluate(G, D, perceptual_loss, writer, 100, 100, motion_features, save_checkpoints=False)
    
