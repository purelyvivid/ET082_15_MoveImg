import tensorflow as tf
from vgg import get_model_VGG, get_model_VGGface
from components import AdaptiveMaxPool2d

# build vgg models
vgg = get_model_VGG() 
vggface = get_model_VGGface()
AdaMaxPool_for_vgg_input = AdaptiveMaxPool2d(output_dim=(224,224),data_format='channels_last')

LOSS_VGG_FACE_WEIGHT = 2e-3
LOSS_VGG19_WEIGHT = 1e-2
LOSS_MCH_WEIGHT = 8e1
l1_loss = lambda x, x_hat : tf.reduce_mean(tf.math.abs(x - x_hat))
#l2_loss = lambda x, x_hat : tf.reduce_mean(tf.math.square(x - x_hat))


def extract_features(inputs, model, layer_idx_list=[0]):
    extracted = []
    x = inputs
    for i, layer in enumerate(vgg.layers):
        x = layer(x)
        if i in layer_idx_list:
            extracted.append(x)
            #print(i, x.shape)
        if len(extracted) == len(layer_idx_list):
            break
    return extracted 

def extract_features_and_calculate_loss(x, x_hat, 
                                        model, 
                                        layer_idx_list=[0], 
                                        loss_fn=l1_loss):
    
    extracted_x = extract_features(x, model, layer_idx_list)
    extracted_x_hat = extract_features(x_hat, model, layer_idx_list)
    
    loss_list = []
    for features_x, features_x_hat in zip(extracted_x, extracted_x_hat):
        loss = loss_fn(features_x, features_x_hat)
        loss_list.append(loss)
    return tf.math.reduce_mean(loss_list) 


def loss_cnt(x, x_hat):
    """
    x, x_hat: shape (B, 256, 256, 3)
    """
    assert x.shape == x_hat.shape  # x, x_hat: shape (B, 256, 256, 3)
    x, x_hat = map(AdaMaxPool_for_vgg_input, [x, x_hat]) # x, x_hat: shape (B, 224, 224, 3)

    broadcast = lambda t : tf.broadcast_to(tf.reshape(t, [1, 1, 1, 3]), x.shape)
    
    IMG_NET_MEAN = broadcast(tf.constant([0.485, 0.456, 0.406]))
    IMG_NET_STD = broadcast(tf.constant([0.229, 0.224, 0.225]))
    
    x = (x - IMG_NET_MEAN) / IMG_NET_STD
    x_hat = (x_hat - IMG_NET_MEAN) / IMG_NET_STD
    
    vgg_face_loss = extract_features_and_calculate_loss(x, x_hat, vggface, [1, 6, 11, 18, 25])# VGG Face Loss
    vgg19_loss = extract_features_and_calculate_loss(x, x_hat, vgg, [1, 6, 11, 18, 25])# VGG19 Loss
    
    return LOSS_VGG_FACE_WEIGHT* vgg_face_loss + LOSS_VGG19_WEIGHT*vgg19_loss

def loss_adv(r_x_hat):
    return -tf.math.reduce_mean(r_x_hat)

def loss_mch(e_hat, W_i):
    return l1_loss(W_i, e_hat) * LOSS_MCH_WEIGHT

def get_lossEG(x, x_hat, r_x_hat, e_hat=None, W_i=None, with_match_loss=False):
    cnt = loss_cnt(x, x_hat)
    adv = loss_adv(r_x_hat)
    mch = loss_mch(e_hat, W_i) if with_match_loss else 0  
    return tf.math.reduce_sum([cnt, adv, mch])

def get_lossD(r_x, r_x_hat):
    return tf.math.reduce_sum(tf.nn.relu(1 + r_x_hat) + tf.nn.relu(1 - r_x))

   

