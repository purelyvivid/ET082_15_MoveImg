import tensorflow as tf
from components import ResidualBlock, AdaptiveResidualBlock, ResidualBlockDown, \
AdaptiveResidualBlockUp, SelfAttention, AdaptiveMaxPool2d
from tensorflow_addons.layers import InstanceNormalization
   

class Embedder(tf.keras.Model):
    """
    The Embedder network attempts to generate a vector that encodes 
    the personal characteristics of an individual given
    a head-shot and the matching landmarks.
    """
    def __init__(self, emb_size=512, data_format='channels_last'):
        super().__init__()
        
        tf.keras.backend.set_image_data_format(data_format)
        self.data_format = data_format
        
        self.emb_size = emb_size
        self.conv1 = ResidualBlockDown(64)
        self.conv2 = ResidualBlockDown(128)
        self.conv3 = ResidualBlockDown(256)
        self.att = SelfAttention(256)
        self.conv4 = ResidualBlockDown(512)
        self.conv5 = ResidualBlockDown(512)

        self.pooling = AdaptiveMaxPool2d((1, 1))
 
    def call(self, x, y): # x: image , y: landmark
        """
        x: image    , 4 dim, [BxK, 128, 128, 3]
        y: landmark , 4 dim, [BxK, 128, 128, 3]
        There are B batches, every batch has K frames(the same person)
        """
        assert len(x.shape) == 4 
        assert x.shape == y.shape
        d_channel = 1 if self.data_format=='channels_first' else 3
        assert x.shape[d_channel] == 3

        # Concatenate x & y
        out = tf.concat([x, y], d_channel) # [BxK, 128, 128, 6]  
        # Encode
        out = self.conv1(out)  # [BxK, 64, 64, 64]
        out = self.conv2(out)  # [BxK, 128, 32, 32]
        out = self.conv3(out)  # [BxK, 256,  16, 16]
        out = self.att(out)    # [BxK, 256,  16, 16]
        out = self.conv4(out)  # [BxK, 256,  8, 8]
        out = self.conv5(out)  # [BxK, 512,  4, 4]

        # Vectorize
        out = tf.nn.relu(self.pooling(out))  # [BxK, 512, 1, 1]
        out = tf.reshape(out, (-1, self.emb_size) )
        return out
    
    
class Generator(tf.keras.Model):

    def __init__(self, emb_size=512, data_format='channels_last' ):
        super().__init__()

        tf.keras.backend.set_image_data_format(data_format)
        self.data_format = data_format        
        # encoding layers
        self.conv1 = ResidualBlockDown(64)
        self.in1_e = InstanceNormalization()

        self.conv2 = ResidualBlockDown(128)
        self.in2_e = InstanceNormalization()

        self.conv3 = ResidualBlockDown(256)
        self.in3_e = InstanceNormalization()

        self.att1 = SelfAttention(256)
        
        self.conv4 = ResidualBlockDown(512)
        self.in4_e = InstanceNormalization()
        
        self.conv5 = ResidualBlockDown(512)
        self.in5_e = InstanceNormalization()

        # residual layers
        self.res1 = AdaptiveResidualBlock(512)
        self.res2 = AdaptiveResidualBlock(512)
        self.res3 = AdaptiveResidualBlock(512)
        self.res4 = AdaptiveResidualBlock(512)
        self.res5 = AdaptiveResidualBlock(512)
        
        # decoding layers
        self.deconv5 = AdaptiveResidualBlockUp(512, 512, upsample=2, emb_size=emb_size)
        self.in5_d = InstanceNormalization()
        
        self.deconv4 = AdaptiveResidualBlockUp(512, 256, upsample=2, emb_size=emb_size)
        self.in4_d = InstanceNormalization()

        self.deconv3 = AdaptiveResidualBlockUp(256, 128, upsample=2, emb_size=emb_size)
        self.in3_d = InstanceNormalization()

        self.att2 = SelfAttention(128)

        self.deconv2 = AdaptiveResidualBlockUp(128, 64, upsample=2, emb_size=emb_size)
        self.in2_d = InstanceNormalization()

        self.deconv1 = AdaptiveResidualBlockUp(64, 3, upsample=2, emb_size=emb_size)
        self.in1_d = InstanceNormalization()

        
    def call(self, y, emb, training=True):
        """
        y: landmark    , 4 dim, [B, 128, 128, 3]  
        emb: embedding , 2 dim, [B, 512]  
        """

        out = y  # [B, 3,  128, 128]

        # Encode
        out = self.in1_e(self.conv1(out), training=training )  # [B, 64, 64, 64]
        out = self.in2_e(self.conv2(out), training=training )  # [B, 128, 32, 32]
        out = self.in3_e(self.conv3(out), training=training )  # [B, 256, 16, 16]
        out = self.att1(out)
        out = self.in4_e(self.conv4(out), training=training )  # [B, 512, 8, 8]
        out = self.in5_e(self.conv5(out), training=training )  # [B, 512, 4, 4]
        
        # Residual layers
        out = self.res1(out, emb)
        out = self.res2(out, emb)
        out = self.res3(out, emb)
        out = self.res4(out, emb)
        out = self.res5(out, emb)
        
        # Decode
        out = self.in5_d(self.deconv5(out, emb), training=training )  # [B, 256, 8, 8]
        out = self.in4_d(self.deconv4(out, emb), training=training )  # [B, 128, 16, 16]
        out = self.in3_d(self.deconv3(out, emb), training=training )  # [B, 64, 32, 32]
        out = self.att2(out)
        out = self.in2_d(self.deconv2(out, emb), training=training )  # [B, 3, 64, 64]
        out = self.in1_d(self.deconv1(out, emb), training=training )  # [B, 3, 128, 128]

        out = tf.math.sigmoid(out)

        return out
    
    
class Discriminator(tf.keras.Model):
    def __init__(self, num_class, emb_size=512, data_format='channels_last'):
        """
        num_class: number of video
        """
        super().__init__()
        
        tf.keras.backend.set_image_data_format(data_format)
        self.data_format = data_format
        self.emb_size = emb_size
        self.num_class = num_class

        self.conv1 = ResidualBlockDown(64)
        self.conv2 = ResidualBlockDown(128)
        self.conv3 = ResidualBlockDown(256)
        self.att = SelfAttention(256)
        self.conv4 = ResidualBlockDown(512)
        self.res = ResidualBlock(512)

        self.pooling = AdaptiveMaxPool2d((1, 1))
        
        self.linear = tf.keras.layers.Dense(num_class, input_shape=(self.emb_size,), use_bias=False)
        

    def call(self, x, y, c_categ):
        """
        x: image                    , 4 dim, [B, 128, 128, 3]  
        y: landmark                 , 4 dim, [B, 128, 128, 3]   
        c_categ: true label of class_index, 2 dim, [B, num_classes]
        data_format="channels_frist"
        """
        assert len(x.shape) == 4 
        assert x.shape == y.shape
        d_channel = 1 if self.data_format=='channels_first' else 3
        assert x.shape[d_channel] == 3

        # Concatenate x & y
        out = tf.concat([x, y], d_channel) # [B, 6, 128, 128] 

        # Encode
        out = self.conv1(out)     # [B,  64, 64, 64]
        out = self.conv2(out)     # [B, 128, 32, 32]
        out = self.conv3(out)     # [B, 256, 16, 16]
        out = self.att(out)
        out = self.conv4(out)     # [B, 512, 8, 8]
        out = self.res(out)       # [B, 512, 4, 4]

        # Vectorize
        out = self.pooling(out)  # [B, 512, 1, 1]
        out = tf.reshape(out, (-1, self.emb_size)) # [B, 512]
        
        # Projection (cGANs with Projection Discriminator) : Temp
        assert c_categ.shape[0] == x.shape[0]
        assert c_categ.shape[1] == self.num_class
        out = self.linear(out) # [B, num_classes], NO sigmoid activation
        projection = tf.reduce_sum(c_categ * out, 1, keepdims=True)# [B, 1]
        
        return projection
    
