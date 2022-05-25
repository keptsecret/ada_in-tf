import tensorflow as tf
import tensorflow.keras as K

class AdaIN():
    def __call__(self, content_feats : tf.Tensor, style_feats : tf.Tensor):
        """
        Feature tensors should be in shape N x C x W x H
        """
        c_mean, c_std = self._compute_mean_std(content_feats)
        s_mean, s_std = self._compute_mean_std(style_feats)

        t = (s_std * (content_feats - c_mean) / c_std) + s_mean
        return tf.transpose(t, perm=[0,2,3,1])

    def _compute_mean_std(self, feats : tf.Tensor, eps=1e-8):
        """
        feats: Features should be in shape N x C x W x H
        """
        n = tf.shape(feats)[0]
        c = tf.shape(feats)[1]
        x = tf.reshape(feats, [n, c, -1])
        mean = tf.reshape(tf.reduce_mean(x, axis=-1), [n, c, 1, 1])
        std = tf.reshape(tf.reduce_mean(x, axis=-1), [n, c, 1, 1]) + eps
        return mean, std

class Encoder(K.Model):
    def __init__(self, pretrained=True, trainable=False):
        """
        Uses VGG19
        """
        super().__init__()
        if pretrained:
            w = 'imagenet'
        else:
            w = None
        vgg = K.applications.VGG19(include_top=False, weights=w)

        for l in vgg.layers:
            l.trainable = trainable

        self.input_layer = K.layers.InputLayer(input_shape=[None, None, 3])
        self.block1 = K.Sequential(vgg.layers[:2])
        self.block2 = K.Sequential(vgg.layers[2:5])
        self.block3 = K.Sequential(vgg.layers[5:8])
        self.block4 = K.Sequential(vgg.layers[8:13])

    def call(self, x, return_all=False):
        x = self.input_layer(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = self.block4(x3)
        return (x1, x2, x3, out) if return_all else out

class Decoder(K.Model):
    def __init__(self):
        super().__init__()

        self.block1 = K.Sequential([
            K.layers.Conv2D(256, 3, strides=1, padding='same', input_shape=[None, None, 512]),
            K.layers.ReLU(),
            K.layers.UpSampling2D()
        ])

        self.block2 = K.Sequential([
            K.layers.Conv2D(256, 3, strides=1, padding='same', input_shape=[None, None, 256]),
            K.layers.ReLU(),
            K.layers.Conv2D(256, 3, strides=1, padding='same'),
            K.layers.ReLU(),
            K.layers.Conv2D(256, 3, strides=1, padding='same'),
            K.layers.ReLU(),
            K.layers.Conv2D(128, 3, strides=1, padding='same'),
            K.layers.ReLU(),
            K.layers.UpSampling2D()
        ])

        self.block3 = K.Sequential([
            K.layers.Conv2D(128, 3, strides=1, padding='same', input_shape=[None, None, 128]),
            K.layers.ReLU(),
            K.layers.Conv2D(64, 3, strides=1, padding='same'),
            K.layers.ReLU(),
            K.layers.UpSampling2D()
        ])

        self.block4 = K.Sequential([
            K.layers.Conv2D(64, 3, strides=1, padding='same', input_shape=[None, None, 64]),
            K.layers.ReLU(),
            K.layers.Conv2D(3, 3, strides=1, padding='same')
        ])

    def call(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        out = self.block4(x)
        return out

class StyleNet(K.Model):
    def __init__(self):
        """
        When calling model on inputs
        inputs: a dict with key-value pairs for content images and style images (+ optional alpha value)
        """
        super().__init__()

        self.encoder = Encoder()
        self.adain = AdaIN()
        self.decoder = Decoder()

    def encode(self, x, return_all=True):
        return self.encoder(x, return_all=return_all)

    def call(self, inputs : dict):
        content_imgs = inputs['content_imgs']
        style_imgs = inputs['style_imgs']
        alpha = inputs['alpha']

        content_feats = self.encoder(content_imgs)
        style_feats = self.encoder(style_imgs)

        t = self.adain(tf.transpose(content_feats, perm=[0,3,1,2]), tf.transpose(style_feats, perm=[0,3,1,2]))
        t = alpha * t + (1 - alpha) * content_feats

        out = self.decoder(t)

        return out, t
