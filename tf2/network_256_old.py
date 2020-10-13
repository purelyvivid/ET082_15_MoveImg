import tensorflow as tf
from components import ResidualBlock, AdaptiveResidualBlock, ResidualBlockDown, \
AdaptiveResidualBlockUp, SelfAttention, AdaptiveMaxPool2d
   

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
        self.att = SelfAttention()
        self.conv4 = ResidualBlockDown(512)
        self.conv5 = ResidualBlockDown(512)
        self.conv6 = ResidualBlockDown(512)

        self.pooling = AdaptiveMaxPool2d((1, 1))
 
    def call(self, x, y): # x: image , y: landmark
        """
        x: image    , 4 dim, [BxK, 3, 256, 256]  or [BxK, 256, 256, 3]
        y: landmark , 4 dim, [BxK, 3, 256, 256]  or [BxK, 256, 256, 3]
        data_format="channels_frist"
        There are B batches, every batch has K frames(the same person)
        """
        assert len(x.shape) == 4 
        assert x.shape == y.shape
        d_channel = 1 if self.data_format=='channels_first' else 3
        assert x.shape[d_channel] == 3

        # Concatenate x & y
        out = tf.concat([x, y], d_channel) # [BxK, 6, 256, 256]  
        # Encode
        out = self.conv1(out)  # [BxK, 64, 128, 128]
        out = self.conv2(out)  # [BxK, 128, 64, 64]
        out = self.conv3(out)  # [BxK, 256, 32, 32]
        out = self.att(out)
        out = self.conv4(out)  # [BxK, 512, 16, 16]
        out = self.conv5(out)  # [BxK, 512, 8, 8]
        out = self.conv6(out)  # [BxK, 512, 4, 4]

        # Vectorize
        out = tf.nn.relu(self.pooling(out))  # [BxK, 512, 1, 1]
        out = tf.reshape(out, (-1, self.emb_size) )
        return out
    
    def trainable_variables(self):
        return [ v for v in self.variables if v.trainable]
    
class Generator(tf.keras.Model):
    """
    Note: tf2 has NO InstanceNorm2d, use BatchNormalization (batch_size=1) to replace
    """

    def __init__(self, emb_size=512, data_format='channels_last' ):
        super().__init__()

        tf.keras.backend.set_image_data_format(data_format)
        self.data_format = data_format        
        # encoding layers
        self.conv1 = ResidualBlockDown(64)
        self.in1_e = tf.keras.layers.BatchNormalization()

        self.conv2 = ResidualBlockDown(128)
        self.in2_e = tf.keras.layers.BatchNormalization()

        self.conv3 = ResidualBlockDown(256)
        self.in3_e = tf.keras.layers.BatchNormalization()

        self.att1 = SelfAttention()

        self.conv4 = ResidualBlockDown(512)
        self.in4_e = tf.keras.layers.BatchNormalization()

        self.conv5 = ResidualBlockDown(512)
        self.in5_e = tf.keras.layers.BatchNormalization()

        self.conv6 = ResidualBlockDown(512)
        self.in6_e = tf.keras.layers.BatchNormalization()

        # residual layers
        self.res1 = AdaptiveResidualBlock(512)
        self.res2 = AdaptiveResidualBlock(512)
        self.res3 = AdaptiveResidualBlock(512)
        self.res4 = AdaptiveResidualBlock(512)
        self.res5 = AdaptiveResidualBlock(512)

        # decoding layers
        self.deconv6 = AdaptiveResidualBlockUp(512, 512, upsample=2, emb_size=emb_size)
        self.in6_d = tf.keras.layers.BatchNormalization()

        self.deconv5 = AdaptiveResidualBlockUp(512, 512, upsample=2, emb_size=emb_size)
        self.in5_d = tf.keras.layers.BatchNormalization()

        self.deconv4 = AdaptiveResidualBlockUp(512, 256, upsample=2, emb_size=emb_size)
        self.in4_d = tf.keras.layers.BatchNormalization()

        self.deconv3 = AdaptiveResidualBlockUp(256, 128, upsample=2, emb_size=emb_size)
        self.in3_d = tf.keras.layers.BatchNormalization()

        self.att2 = SelfAttention()

        self.deconv2 = AdaptiveResidualBlockUp(128, 64, upsample=2, emb_size=emb_size)
        self.in2_d = tf.keras.layers.BatchNormalization()

        self.deconv1 = AdaptiveResidualBlockUp(64, 3, upsample=2, emb_size=emb_size)
        self.in1_d = tf.keras.layers.BatchNormalization()

        
    def call(self, y, emb, training=True):
        """
        y: landmark    , 4 dim, [B, 3, 256, 256]  
        emb: embedding , 2 dim, [B, 512]  
        data_format="channels_frist"
        """

        out = y  # [B, 3, 256, 256]

        # Encode
        out = self.in1_e(self.conv1(out), training=training )  # [B, 64, 128, 128]
        out = self.in2_e(self.conv2(out), training=training )  # [B, 128, 64, 64]
        out = self.in3_e(self.conv3(out), training=training )  # [B, 256, 32, 32]
        out = self.att1(out)
        out = self.in4_e(self.conv4(out), training=training )  # [B, 512, 16, 16]
        out = self.in5_e(self.conv5(out), training=training )  # [B, 512, 8, 8]
        out = self.in6_e(self.conv6(out), training=training )  # [B, 512, 4, 4]

        # Residual layers
        out = self.res1(out, emb)
        out = self.res2(out, emb)
        out = self.res3(out, emb)
        out = self.res4(out, emb)
        out = self.res5(out, emb)

        # Decode
        out = self.in6_d(self.deconv6(out, emb), training=training )  # [B, 512, 4, 4]
        out = self.in5_d(self.deconv5(out, emb), training=training )  # [B, 512, 16, 16]
        out = self.in4_d(self.deconv4(out, emb), training=training )  # [B, 256, 32, 32]
        out = self.in3_d(self.deconv3(out, emb), training=training )  # [B, 128, 64, 64]
        out = self.att2(out)
        out = self.in2_d(self.deconv2(out, emb), training=training )  # [B, 64, 128, 128]
        out = self.in1_d(self.deconv1(out, emb), training=training )  # [B, 3, 256, 256]

        out = tf.math.sigmoid(out)

        return out
    
    def trainable_variables(self):
        return [ v for v in self.variables if v.trainable]
    
class Discriminator(tf.keras.Model):
    def __init__(self, num_class, data_format='channels_last'):
        """
        num_class: number of people
        """
        super().__init__()
        
        tf.keras.backend.set_image_data_format(data_format)
        self.data_format = data_format
        
        self.num_class = num_class

        self.conv1 = ResidualBlockDown(64)
        self.conv2 = ResidualBlockDown(128)
        self.conv3 = ResidualBlockDown(256)
        self.att = SelfAttention()
        self.conv4 = ResidualBlockDown(512)
        self.conv5 = ResidualBlockDown(512)
        self.conv6 = ResidualBlockDown(512)
        self.res_block = ResidualBlock(512)

        self.pooling = AdaptiveMaxPool2d((1, 1))
        
        self.linear = tf.keras.layers.Dense(num_class, input_shape=(512,), use_bias=False)
        

    def call(self, x, y, c):
        """
        x: image                    , 4 dim, [B, 3, 256, 256]  
        y: landmark                 , 4 dim, [B, 3, 256, 256]   
        c: true label of class_index, 2 dim, [B, 1] , value in {0, ..., num_class}
        data_format="channels_frist"
        """
        assert len(x.shape) == 4 
        assert x.shape == y.shape
        d_channel = 1 if self.data_format=='channels_first' else 3
        assert x.shape[d_channel] == 3

        # Concatenate x & y
        out = tf.concat([x, y], d_channel) # [B, 6, 256, 256] 

        # Encode
        out_0 = self.conv1(out)       # [B, 64, 128, 128]
        out_1 = self.conv2(out_0)     # [B, 128, 64, 64]
        out_2 = self.conv3(out_1)     # [B, 256, 32, 32]
        out_3 = self.att(out_2)
        out_4 = self.conv4(out_3)     # [B, 512, 16, 16]
        out_5 = self.conv5(out_4)     # [B, 512, 8, 8]
        out_6 = self.conv6(out_5)     # [B, 512, 4, 4]
        out_7 = self.res_block(out_6) # [B, 512, 4, 4]

        # Vectorize
        out = self.pooling(out_7)  # [B, 512, 1, 1]
        out = tf.reshape(out, (-1, 512)) # [B, 512]
        
        # Projection (cGANs with Projection Discriminator) : Temp
        c = tf.reshape(c, (x.shape[0],1)) # [B, 1]
        c_categ = tf.keras.utils.to_categorical(c, num_classes=self.num_class) # [B, num_classes]
        
        out = self.linear(out) # [B, num_classes], NO sigmoid activation
        
        projection = tf.reduce_sum(c_categ * out, 1, keepdims=True)# [B, 1]
        
        return projection
    
    def trainable_variables(self):
        return [ v for v in self.variables if v.trainable]