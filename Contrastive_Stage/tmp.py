import numpy as np

path='/Users/jinbin/5340Proj/dataset/train/0/motion.npy'

#load the motion data and print first 1 frame
motion = np.load('/Users/jinbin/5340Proj/dataset/train/0/motion.npy')

print(motion[0, :,:])