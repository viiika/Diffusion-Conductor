import os
import torch
import numpy as np
import argparse
from os.path import join as pjoin

from utils.plot_script import *
from utils.get_opt import get_opt

from trainers import DDPMTrainer
from models import MotionTransformer
from utils.utils import *
import librosa
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip

from scipy.signal import savgol_filter
import tqdm

def smooth_motion(kp_pred, kernel=11, order=5):
    for i in range(kp_pred.shape[1]):
        for j in range(2):
            data = kp_pred[:, i, j]
            data_smooth = savgol_filter(data, kernel, order)
            kp_pred[:, i, j] = data_smooth
    return kp_pred

def vis_img(img, kp_preds, kp_scores, hand_trace):
    """
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii
    return rendered image
    """

    ''' --- MOCO keypoints ---
    0 Nose, 1 LEye, 2 REye, 3 LEar, 4 REar
    5 LShoulder, 6 RShoulder, 7 LElbow, 8 RElbow, 9 LWrist, 10 RWrist
    11 LHip, 12 RHip, 
    (discarded: 13 LKnee, 14 Rknee, 15 LAnkle, 16 RAnkle, 17 Neck)
    '''
    kp_num = 17
    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (17, 11), (17, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16),  # Legs
        # add body outline:
        (11, 12), (5, 11), (6, 12)
    ]
    black = (120, 120, 120)
    blue = (255, 128, 0)
    red = (51, 41, 255)
    white = (255, 255, 255)
    line_color = [blue, blue, blue, blue,
                  blue, blue, blue, blue, blue,
                  black, black, black, black, black, black,
                  blue, blue, blue  # cdl add body outline
                  ]

    trace_head = np.array(red)
    trace_end = np.array(white)
    for i in range(len(hand_trace)):
        alpha = i / len(hand_trace)

        color = alpha * trace_head + (1 - alpha) * trace_end
        for j in range(len(hand_trace[i])):
            color_factor = i / len(hand_trace)
            cv2.circle(img, (int(hand_trace[i, j, 0]), int(hand_trace[i, j, 1])), 2, color, 2)

    part_line = {}
    # --- Draw points --- #
    vis_thres = 0.4
    for n in range(kp_scores.shape[0]):
        if kp_scores[n] <= vis_thres:
            continue
        cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
        part_line[n] = (int(cor_x), int(cor_y))

    # --- Draw limbs --- #
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            X = (start_xy[0], end_xy[0])
            Y = (start_xy[1], end_xy[1])
            mX = np.mean(X)
            mY = np.mean(Y)

            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), int(stickwidth)), int(angle), 0, 360, 10)
            if i < len(line_color):
                cv2.fillConvexPoly(img, polygon, line_color[i])
            else:
                cv2.line(img, start_xy, end_xy, (128, 128, 128), 1)

    for n in [9, 10]:
        cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
        cv2.circle(img, (int(cor_x), int(cor_y)), 9, white, 9)
        cv2.circle(img, (int(cor_x), int(cor_y)), 2, red, 2)
        cv2.circle(img, (int(cor_x), int(cor_y)), 10, red, 2)

    return img

def vis_motion(motions, kp_score=None, save_path='../test/result', name='_[name]_', post_processing=True):
    motions = np.array(motions)
    motions = motions.reshape([motions.shape[0],motions.shape[1],13,2])

    # motions [num_conductor, num_frame, 13, 2]
    if kp_score is None:  # confidence
        kp_score = np.zeros((motions[0].shape[0], 17))
        kp_score[:, :13] = 1

    window = 600
    trace_len = 30
    hand_traces = []
    video_file = save_path + name + '.avi'
    wirter = cv2.VideoWriter(video_file, 0, cv2.VideoWriter_fourcc(*'XVID'), 30,
                             (1 + len(motions) * window, window))

    for i in range(len(motions)):
        motions[i] *= window
        motion = motions[i]
        motion = smooth_motion(motion, kernel=19)
        hand_trace = np.ones((motion.shape[0] + trace_len, 2, 2)) * -1
        hand_trace[trace_len:, :, :] = motion[:, 9:11, :]
        hand_traces.append(hand_trace)

    for f in tqdm.tqdm(range(motions[0].shape[0])):
        frame = np.ones((window, 1, 3), np.uint8) * 255
        for i in range(len(motions)):
            motion = motions[i]
            background = np.ones((window, window, 3), np.uint8) * 255
            img = vis_img(background, motion[f], kp_score[f], hand_traces[i][f:f + trace_len, :, :])
            frame = np.concatenate((frame, img), axis=1)

        wirter.write(frame)
    wirter.release()
    cv2.destroyAllWindows()
    return video_file

def plot_music2motion(data, result_path, npy_path, title):
    saved_video_file = vis_motion([data], save_path=result_path, name=title)

    video = VideoFileClip(saved_video_file)
    test_samles_dir = '/home/zhuoran/DiffuseConductor/assets/music/Beethoven Symphony 7.mp3' 
    video = video.set_audio(AudioFileClip(test_samles_dir))
    video.write_videofile(saved_video_file + '.mp4')

def extract_mel_feature(audio_file, mel_len_90fps=None):
    y, sr = librosa.load(audio_file)

    # select only first 60s
    if len(y) > sr * 60:
        y = y[:sr * 60]

    if mel_len_90fps is None:
        mel_len_90fps = int(len(y) / sr * 90)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=256)
    mel_dB = librosa.power_to_db(mel, ref=np.max)

    norm_mel = np.flip(np.abs(mel_dB + 80) / 80, 0)
    resized_mel = cv2.resize(norm_mel, (mel_len_90fps, norm_mel.shape[0]))
    return resized_mel.T

def build_models(opt):
    encoder = MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        device = opt.device,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, default="/home/zhuoran/DiffuseConductor/Diffusion_Stage/checkpoints/ConductorMotion100/train/opt.txt", help='Opt path')
    parser.add_argument('--music_path', type=str, default="/home/zhuoran/DiffuseConductor/assets/music/Beethoven Symphony 7.mp3", help='Music Path for motion generation')
    parser.add_argument('--motion_length', type=int, default=60, help='Number of frames for motion generation')
    parser.add_argument('--result_path', type=str, default="test_sample.gif", help='Path to save generation result')
    parser.add_argument('--npy_path', type=str, default="", help='Path to save 3D keypoints sequence')
    parser.add_argument('--gpu_id', type=int, default=5, help="which gpu to use")
    args = parser.parse_args()
    
    device = torch.device('cuda:%d' % args.gpu_id if args.gpu_id != -1 else 'cpu')

    opt = get_opt(args.opt_path, device)
    opt.do_denoise = True

    assert args.motion_length <= 196

    opt.joints_num = 13
    opt.dim_pose = 26

    encoder = build_models(opt).to(device)
    
    trainer = DDPMTrainer(opt, encoder)
    trainer.load(pjoin(opt.model_dir, 'latest.tar'))
    # trainer.load(pjoin(opt.model_dir, 'ckpt_e475.tar'))

    trainer.eval_mode()
    trainer.to(opt.device)

    result_dict = {}

    with torch.no_grad():
        if args.motion_length != -1:
            music_path = args.music_path
            mel = extract_mel_feature(music_path)
            
            pred_motions = trainer.generate_music_motion(mel, opt.dim_pose)
            motion = pred_motions[0].cpu().numpy()
            motion = motion.reshape([motion.shape[0],13,2])

            # motion = motion * std + mean
            title = " #%d" % motion.shape[0]
            
            plot_music2motion(motion, args.result_path, args.npy_path, title)
