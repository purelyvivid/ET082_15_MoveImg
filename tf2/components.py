import tensorflow as tf
from spectral_norm import SpectralNormalization
from tensorflow_addons.layers import InstanceNormalization

class ConvLayer(tf.keras.layers.Layer):

    def __init__(self, out_channels, kernel_size, stride, padding=None, data_format="channels_last"):
        super().__init__()
        self.data_format = data_format 
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv2d = SpectralNormalization(
            tf.keras.layers.Conv2D(filters=out_channels,
                                   kernel_size=kernel_size,
                                   strides=stride,
                                   kernel_initializer="truncated_normal",#weights_init,
                                   data_format=data_format
                                   )) 
        
    
    def call(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
    
    def reflection_pad(self, x):
        """
        paddings must be less than the dimension size
        """
        if self.data_format == "channels_first":
            d_C, d_H, d_W = 1,2,3
        elif self.data_format == "channels_last":
            d_C, d_H, d_W = 3,1,2
            
        B, C, H, W = x.shape[0], x.shape[d_C], x.shape[d_H], x.shape[d_W]
        if H == 1 or W == 1:
            return x
            
        if self.padding is None:
            padding = self.kernel_size // 2 # "same" padding
        else:
            padding = self.padding 
            
        if padding==0 or (not padding<H) or (not padding<W): 
            return x
        
        padding_list = [[0,0]]*4
        padding_list[d_H] = [padding, padding]
        padding_list[d_W] = [padding, padding]
        paddings = tf.constant(padding_list)
        return tf.pad(x, paddings, "REFLECT")
    
class SelfAttention(tf.keras.layers.Layer):
    
    def __init__(self, channels, data_format="channels_last"):
        super().__init__()
        self.data_format = data_format 
        assert(channels>8), "channels has to larger than 8!"
        self.q = ConvLayer(channels//8, 1, 1, 0)
        self.k = ConvLayer(channels//8, 1, 1, 0)
        self.v = ConvLayer(channels,    1, 1, 0)
        
        self.gamma = tf.Variable(shape=(1,), initial_value=[0.])
    
    def call(self, x): #BxHxWxC
        """
        inputs: 
            x with shape: (B, H, W, C)
        outputs:
            out with shape: (B, H, W, C)
        """
        
        if self.data_format == "channels_first":
            d_C, d_H, d_W = 1,2,3
        elif self.data_format == "channels_last":
            d_C, d_H, d_W = 3,1,2 # x.shape (B, H, W, C)
        B, C, H, W = x.shape[0], x.shape[d_C], x.shape[d_H], x.shape[d_W]  
        #print("B, C, H, W = ", B, C, H, W)
        
        Q = self.q(x) #BxHxWxC', C'=C//8
        K = self.k(x) #BxHxWxC'
        V = self.v(x) #BxHxWxC
        #print(Q.shape, K.shape, V.shape)
        
        if self.data_format == "channels_last":# change to channel_first for matrix multiply
            trans = lambda t : tf.transpose(t, perm=[0, d_C, d_H, d_W])
            Q = trans(Q) #BxC'xHxW, C'=C//8
            K = trans(K) #BxC'xHxW
            V = trans(V) #BxC xHxW
            
        #print(Q.shape, K.shape, V.shape)
        
        flat = lambda t : tf.reshape(t, shape=(B, -1, H*W) )
        Q = flat(Q) #BxC'xN, N=H*W
        K = flat(K) #BxC'xN
        V = flat(V) #BxC xN
        #print(Q.shape, K.shape, V.shape)
        
        Q_ = tf.transpose(Q, perm=[0, 2, 1])#BxNxC'
        attention = tf.nn.softmax(Q_@K, axis=-1) # (BxNxC') dot (BxC'xN) = BxNxN
        out = V@attention # (BxCxN) dot (BxNxN) = BxCxN
        out = tf.reshape(out, shape=(B, C, H, W) ) # BxCxHxW
        
        if self.data_format == "channels_last":
            out =  tf.transpose(out, perm=[0, 2, 3, 1]) # change back 
        
        gamma = tf.broadcast_to(self.gamma, out.shape)
        out = gamma*out + x

        return out #BxHxWxC

class AdaIn(tf.keras.layers.Layer):
    def __init__(self, data_format="channels_last"):
        super().__init__()
        self.data_format = data_format
        self.eps = 1e-5

    def call(self, x, mean_style, std_style):
        """
        inputs:
            mean_style with shape: (B, C) or (B, C, 1) 
            std_style with shape: (B, C) or (B, C, 1) 
        NOTE: need to broadcast to the same dimension for +/-/*// operation 
        """
        if self.data_format == 'channels_first':
            return self.call_channel_first(x, mean_style, std_style)
        elif self.data_format == "channels_last":
            return self.call_channel_last(x, mean_style, std_style)
        
    def call_channel_last(self, x, mean_style, std_style):
        
        B, H, W, C = x.shape

        feature = tf.reshape(x, (B, -1, C)) # shape: (B, H*W, C)
        
        # feature mean and stddev
        std_feat = tf.math.reduce_std(feature,axis=1) + self.eps # shape: (B, C)
        mean_feat = tf.math.reduce_mean(feature,axis=1)          # shape: (B, C)
        
        # broadcast , before shape: (B, C) or (B, C, 1)  , after shape: (B, H*W, C)
        broadcast = lambda t : tf.broadcast_to(tf.reshape(t, (B, 1, C)), feature.shape)
        std_style = broadcast(std_style)
        mean_style = broadcast(mean_style)
        std_feat = broadcast(std_feat)
        mean_feat = broadcast(mean_feat)

        adain = std_style * (feature - mean_feat) / std_feat + mean_style # shape: (B, H*W, C)

        adain = tf.reshape(adain, (B, H, W, C))
        return adain
        
    def call_channel_first(self, x, mean_style, std_style):
        
        B, C, H, W = x.shape

        feature = tf.reshape(x, (B, C, -1)) # shape: (B, C, H*W)
        
        # feature mean and stddev
        std_feat = tf.math.reduce_std(feature,axis=2) + self.eps # shape: (B, C)
        mean_feat = tf.math.reduce_mean(feature,axis=2)          # shape: (B, C)
        
        # broadcast , before shape: (B, C) or (B, C, 1)  , after shape: (B, C, H*W)
        broadcast = lambda t : tf.broadcast_to(tf.reshape(t, (B, C, 1)), feature.shape)
        std_style = broadcast(std_style)
        mean_style = broadcast(mean_style)
        std_feat = broadcast(std_feat)
        mean_feat = broadcast(mean_feat)

        adain = std_style * (feature - mean_feat) / std_feat + mean_style # shape: (B, C, H*W)

        adain = tf.reshape(adain, (B, C, H, W))
        return adain
    
class ResidualBlockDown(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()

        # Right Side
        self.conv_r1 = ConvLayer(out_channels, kernel_size, stride, padding)
        self.conv_r2 = ConvLayer(out_channels, kernel_size, stride, padding)

        # Left Side
        self.conv_l = ConvLayer(out_channels, 1, 1)

    
    def call(self, x):
        residual = x

        # Right Side
        out = tf.nn.relu(x)
        out = self.conv_r1(out)
        out = tf.nn.relu(out)
        out = self.conv_r2(out)
        out = tf.keras.backend.pool2d(out, (2,2), strides=(2,2))

        # Left Side
        residual = self.conv_l(residual)
        residual = tf.keras.backend.pool2d(residual, (2,2), strides=(2,2))

        # Merge
        out = residual + out
        return out
    
    
    
class ResidualBlockUp(tf.keras.layers.Layer):

    def __init__(self, out_channels, kernel_size=3, stride=1, upsample=2):
        super().__init__()

        # General
        self.upsample = tf.keras.layers.UpSampling2D(size=(upsample, upsample))

        # Right Side
        self.norm_r1 = InstanceNormalization()
        self.conv_r1 = ConvLayer(out_channels, kernel_size, stride)

        self.norm_r2 = InstanceNormalization()
        self.conv_r2 = ConvLayer(out_channels, kernel_size, stride)

        # Left Side
        self.conv_l = ConvLayer(out_channels, 1, 1)

    def call(self, x, training=True):
        residual = x

        # Right Side
        out = self.norm_r1(x, training=training)
        out = tf.nn.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.norm_r2(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv_r2(out)

        # Left Side
        residual = self.upsample(residual)
        residual = self.conv_l(residual)

        # Merge
        out = residual + out
        return out
    
class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, kernel_size=3, stride=1)
        self.in1 = InstanceNormalization()
        self.conv2 = ConvLayer(channels, kernel_size=3, stride=1)
        self.in2 = InstanceNormalization()

    def call(self, x, training=True):
        residual = x

        out = self.conv1(x)
        out = self.in1(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.in2(out, training=training)

        out = out + residual
        return out
    
class AdaptiveResidualBlockUp(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2, emb_size=512, ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # General
        self.upsample = tf.keras.layers.UpSampling2D(size=(upsample, upsample))

        # Right Side
        self.norm_r1 = AdaIn()
        self.conv_r1 = ConvLayer(out_channels, kernel_size, stride)

        self.norm_r2 = AdaIn()
        self.conv_r2 = ConvLayer(out_channels, kernel_size, stride)

        # Left Side
        self.conv_l = ConvLayer(out_channels, 1, 1)
        
        # for Adaptive ADAIN - projection layer
        self.linear1 = tf.keras.layers.Dense(in_channels*2, input_shape=(emb_size,), use_bias=False)
        self.linear2 = tf.keras.layers.Dense(out_channels*2, input_shape=(emb_size,), use_bias=False)    

    def call(self, x, emb): # emb: fixed size= 512
        
        mean1, std1 = tf.split(self.linear1(emb), num_or_size_splits=2, axis=1)
        mean2, std2 = tf.split(self.linear2(emb), num_or_size_splits=2, axis=1)
        
        residual = x

        # Right Side
        out = self.norm_r1(x, mean1, std1)
        out = tf.nn.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.norm_r2(out, mean2, std2)
        out = tf.nn.relu(out)
        out = self.conv_r2(out)

        # Left Side
        residual = self.upsample(residual)
        residual = self.conv_l(residual)

        # Merge
        out = residual + out
        return out

class AdaptiveResidualBlock(tf.keras.layers.Layer):
    
    def __init__(self, channels, emb_size=512, ):
        super().__init__()
        
        self.conv1 = ConvLayer(channels, kernel_size=3, stride=1)
        self.in1 = AdaIn()
        self.conv2 = ConvLayer(channels, kernel_size=3, stride=1)
        self.in2 = AdaIn()
        
        # for Adaptive ADAIN - projection layer
        self.linear1 = tf.keras.layers.Dense(channels*2, input_shape=(emb_size,), use_bias=False)
        self.linear2 = tf.keras.layers.Dense(channels*2, input_shape=(emb_size,), use_bias=False)

    def call(self, x, emb):# emb: fixed size= 512
        
        mean1, std1 = tf.split(self.linear1(emb), num_or_size_splits=2, axis=1)
        mean2, std2 = tf.split(self.linear2(emb), num_or_size_splits=2, axis=1)
        
        residual = x

        out = self.conv1(x)
        out = self.in1(out, mean1, std1)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.in2(out, mean2, std2)

        out = out + residual
        return out
    
class AdaptiveMaxPool2d(tf.keras.layers.Layer):
    """
    pytorch has nn.AdaptiveMaxPool2d
    but tensorflow 2.0 has not.
    Implement by myself.
    """
    def __init__(self, output_dim=(1,1), data_format='channels_last' ):
        super().__init__()
        self.data_format = data_format
        self.output_dim = output_dim
        self.pool = tf.keras.layers.MaxPool2D(padding='valid', data_format=data_format)
        
    def call(self, x):
        if self.data_format == 'channels_first'  :
            return self.call_channel_first(x)
        elif self.data_format == "channels_last":
            return self.call_channel_last(x)

    
    def call_channel_first(self, x):
        in_B, in_C, in_H, in_W = x.shape 
        op_H, op_W = self.output_dim
        p1_H, p2_H = self.get_paddings_for_outputsize_fully_devided(in_H, op_H)
        p1_W, p2_W = self.get_paddings_for_outputsize_fully_devided(in_W, op_W)
        if p1_H!=0 or p2_H!=0 or p1_W!=0 or p2_W!=0:
            paddings = tf.constant([[0, 0,], [0, 0,], [p1_H, p2_H], [p1_W, p2_W]])
            x = tf.pad(x, paddings, "REFLECT")
            in_B, in_C, in_H, in_W = x.shape
        assert in_H%op_H==0
        assert in_W%op_W==0
        kernel_size_H = self.get_pooling_kernel_size(in_H, op_H)
        kernel_size_W = self.get_pooling_kernel_size(in_W, op_W)
        self.pool.pool_size = (kernel_size_H, kernel_size_W)
        self.pool.strides = (kernel_size_H, kernel_size_W)
        x = self.pool(x)
        assert x.shape[2] == op_H 
        assert x.shape[3] == op_W  
        return x
    
    def call_channel_last(self, x):
        in_B, in_H, in_W, in_C = x.shape 
        op_H, op_W = self.output_dim
        p1_H, p2_H = self.get_paddings_for_outputsize_fully_devided(in_H, op_H)
        p1_W, p2_W = self.get_paddings_for_outputsize_fully_devided(in_W, op_W)
        if p1_H!=0 or p2_H!=0 or p1_W!=0 or p2_W!=0:
            paddings = tf.constant([[0, 0,], [p1_H, p2_H], [p1_W, p2_W], [0, 0,] ])
            x = tf.pad(x, paddings, "REFLECT")
            in_B, in_H, in_W, in_C = x.shape
        assert in_H%op_H==0
        assert in_W%op_W==0
        kernel_size_H = self.get_pooling_kernel_size(in_H, op_H)
        kernel_size_W = self.get_pooling_kernel_size(in_W, op_W)
        self.pool.pool_size = (kernel_size_H, kernel_size_W)
        self.pool.strides = (kernel_size_H, kernel_size_W)
        x = self.pool(x)
        assert x.shape[1] == op_H 
        assert x.shape[2] == op_W  
        return x
        
    def get_pooling_kernel_size(self, in_size, out_size):
        assert in_size%out_size==0
        stride = in_size//out_size
        kernel_size = in_size - (out_size-1)*stride 
        return kernel_size   
        
    def get_paddings_for_outputsize_fully_devided(self, in_size, out_size):
        rem = in_size%out_size
        if rem >0:
            p2 = int((out_size-rem)/2)
            p1 = p2 if rem%2==0 else p2+1
            return p1, p2
        else:
            return 0,0     