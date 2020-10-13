import glob
import numpy as np
import tensorflow as tf

class VoxCelebDataset():
    def __init__(self, root , K=8, img_size=256, data_format='channels_last'):
        super().__init__()
        self.filenames = glob.glob(root + '*.npy')
        self.n_frames = K+1
        self.img_size = img_size  
        self.data_format = data_format
        self.channels_first = data_format=="channels_first"
    
    def __getitem__(self, file_idx):
        
        # get data from a file
        filename = self.filenames[file_idx]
        data = np.load(filename, allow_pickle=True)
        frames = data.item()['frames']
        landmarks = data.item()['landmarks']  
        # sample index of frames
        rand_idx = np.random.randint(0, frames.shape[0], self.n_frames)
        frames = self.to_tensor_n_resize(frames[rand_idx])       # [K+1, 3, img_size, img_size]
        landmarks = self.to_tensor_n_resize(landmarks[rand_idx]) # [K+1, 3, img_size, img_size]

        source_landmark = landmarks[0] # [3, img_size, img_size]
        target_frame = frames[0]       # [3, img_size, img_size]

        x = frames[1:]    # [K, 3, img_size, img_size]
        y = landmarks[1:] # [K, 3, img_size, img_size]

        source_landmark, target_frame, x, y = map(self.scale, [source_landmark, target_frame, x, y])

        return file_idx, source_landmark, target_frame, x, y
    
    def __len__(self, ):    
        return len(self.filenames)
        
    def to_tensor_n_resize(self , x):
        x = tf.image.resize(x, (self.img_size,self.img_size))
        if self.channels_first: x = self.transpose_channels_last2first(x)
        return x
    
    def to_numpy_or_dim3(self , x, 
                         return_numpy=False,
                         return_dim3=False,
                        ):
        if len(x.shape)==3: x = x[None,...] 
        assert len(x.shape)==4
        if return_dim3 and x.shape[0]==1: x = x[0]
        return x.numpy() if return_numpy else x        
        
    def transpose_channels_last2first(self , x, 
                                     return_numpy=False,
                                     return_dim3=False,
                                    ): 
        if len(x.shape)==3: x = x[None,...] 
        assert len(x.shape)==4
        x = tf.transpose(x, [0, 3, 2, 1]) 
        return self.to_numpy_or_dim3(x, return_numpy=return_numpy, return_dim3=return_dim3)
    
    def transpose_channels_first2last(self , x, 
                                     return_numpy=False,
                                     return_dim3=False,
                                    ): 
        if len(x.shape)==3: x = x[None,...] 
        assert len(x.shape)==4
        x = tf.transpose(x, [0, 3, 2, 1]) 
        return self.to_numpy_or_dim3(x, return_numpy=return_numpy, return_dim3=return_dim3)
  
    def scale(self , x):
        return x/255.0 
    
    def transpose_for_plotting(self, x):
        if self.data_format=="channels_first":
            return self.transpose_channels_first2last(x, return_numpy=True, return_dim3=True)
        else:
            return self.to_numpy_or_dim3(x, return_numpy=True, return_dim3=True)
        
    
def DatasetGen(dataset, batch_size=4):
    file_idx = 0
    while True:
        #print(f"file_idx:{file_idx}")
        yield dataset[file_idx]
        file_idx += 1
        if file_idx == batch_size: file_idx = 0
        
        
"""
# test
root = "../few-show/data_npy/"
dataset = VoxCelebDataset(root)
datagen = DatasetGen(dataset)

file_idx, source_landmark, target_frame, x, y = next(iter(datagen))
print(file_idx)
file_idx, source_landmark, target_frame, x, y = next(iter(datagen))
print(file_idx)
file_idx, source_landmark, target_frame, x, y = next(iter(datagen))
print(file_idx)
file_idx, source_landmark, target_frame, x, y = next(iter(datagen))
print(file_idx)
"""