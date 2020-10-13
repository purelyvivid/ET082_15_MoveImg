# ET082_15_MoveImg

|  |  |  |  |
| -------- | -------- | -------- | -------- |
| Original | <img src="./result/ko.jpg" width="90" height="120" />     | <img src="./result/han.jpg" width="90" height="120" />     | <img src="./result/sun.jpg" width="90" height="120" />     |
| Moving | <img src="./result/ko_299.gif"  width="90" height="120" />     | <img src="./result/han_299.gif" width="90" height="120" />     | <img src="./result/sun_299.gif" width="90" height="120" />     |
| 
|Original|<img src="./result/in.jpg" width="90" height="120" />|<img src="./result/van.jpg" width="90" height="120" />|<img src="./result/mona_bw_.jpg" width="90" height="120" />|
| Moving |<img src="./result/in_299.gif" width="90" height="120" />|<img src="./result/van_299.gif" width="90" height="120" />|<img src="./result/mona_bw_299_light.gif" width="90" height="120" />|


## File Structure

### / tf2
- 2019/10/16
- data preprocessing
- try tensorflow 2.0 for modling

### / torch
- 2019/11/2
- training for meta stage

### / finetune
- 2019/11/15
- training for fine-tune stage
- generate `.gif` result (moving image)

### / git
- provide code for git pull/push to Gitlab via SSH key

### / result
- put some result for demo 


## Code Structure

### Data Source
- VoxCeleb1 dataset
- Raw data from link: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- or download from http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/data/dense-face-frames.tar.gz (~27GB)

### Data Preprocessing
- / tf2 / 1-download_n_process_data.ipynb

### Model with Tensorflow 2.0 (not train well)
- / tf2 / 2-build_model_n_train.ipynb
- Generated image_size: 128 x 128

### Model with PyTorch
- / torch / train_256.ipynb
- Generated image_size: 256 x 256

### FineTune with PyTorch
- / finetune / finetune.ipynb
- Generated `.gif` file (image_size: 256 x 256)

## Front-end Deployment
Download from: https://drive.google.com/open?id=1dwEdRyOJDopomCFiHBpFWa3mVIaA00Nz


## Reference

- paper: [Few-Shot Adversarial Learning of Realistic Neural Talking Head Models](https://arxiv.org/pdf/1905.08233.pdf)

- code: 
    1. https://github.com/grey-eye/talking-heads
    2. https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models
    3. https://github.com/shoutOutYangJie/Few-Shot-Adversarial-Learning-for-face-swap