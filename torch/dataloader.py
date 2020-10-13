import glob
import numpy as np
from random import sample
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset

class VoxCelebDataset(Dataset):
    
    def __init__(self, root , K=8, size=128):
        self.clips_npz = glob.glob(root + '*.npy')
        self.K = K+1
        self.size = size
        
    def __len__(self):
        return len(self.clips_npz)
    
    def __getitem__(self, idx):
        f = self.clips_npz[idx]
        data = np.load(f, allow_pickle=True)
        frames = data.item()['frames']
        landmarks = data.item()['landmarks']
        
        rand_idx = sample(range(0,frames.shape[0]) , self.K)
        frames = self.to_tensor_resize(frames[rand_idx])
        landmarks = self.to_tensor_resize(landmarks[rand_idx])
        
        source_landmark = landmarks[0]
        target_frames = frames[0]
        
        context = self.combine_(frames[1:] , landmarks[1:])
        
        return idx , source_landmark/255 , target_frames/255 , context/255
    
    def to_tensor_resize(self , x):

        x = torch.from_numpy(x.transpose(0,3,1,2)).float()
        return interpolate(x , size=(self.size ,self.size))
    
    def combine_(self , f , l):
        return torch.cat([f , l] , 1)