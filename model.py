import tensorflow as tf
import tensorflow.keras as K

class AdaIN():
    def __call__(self, content_feats : tf.Tensor, style_feats : tf.Tensor):
        """
        Feature tensors should be in shape N x H x W x C
        """
        c_mean, c_std = self._compute_mean_std(content_feats)
        s_mean, s_std = self._compute_mean_std(style_feats)

        return (s_std * (content_feats - c_mean) / c_std) + s_mean

    def _compute_mean_std(self, feats : tf.Tensor, eps=1e-8):
        """
        feats: Features should be in shape N x H x W x C
        """
        mean = tf.math.reduce_mean(feats, axis=[1,2], keepdims=True)
        std = tf.math.reduce_std(feats, axis=[1,2], keepdims=True) + eps
        return mean, std

class Encoder(K.Model):
    def __init__(self, pretrained=True, trainable_layers=list()):
        """
        Uses VGG19
        """
        super().__init__()
        if pretrained:
            w = 'imagenet'
        else:
            w = None
        vgg = K.applications.VGG19(include_top=False, weights=w)

        self.input_layer = K.layers.InputLayer(input_shape=[None, None, 3])
        self.block1 = K.Sequential(vgg.layers[:2])
        self.block2 = K.Sequential(vgg.layers[2:5])
        self.block3 = K.Sequential(vgg.layers[5:8])
        self.block4 = K.Sequential(vgg.layers[8:13])

        blocks = [self.block1, self.block2, self.block3, self.block4]

        for i, b in enumerate(blocks):
            if i + 1 in trainable_layers:
                for l in b.layers:
                    l.trainable = True
            else:
                for l in b.layers:
                    l.trainable = False

    def call(self, inputs):
        x = inputs['x']
        return_all = inputs['return_all']

        x = self.input_layer(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = self.block4(x3)
        return (x1, x2, x3, out) if return_all else out

class Decoder(K.Model):
    def __init__(self):
        super().__init__()

        # relu_alpha = 1e-3
        self.reflect_padding = K.layers.Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], 'REFLECT'))
        self.upsample = K.layers.UpSampling2D(interpolation='nearest')

        self.block1 = K.Sequential([
            K.layers.Conv2D(256, 3, strides=1, padding='valid', input_shape=[None, None, 512]),
            K.layers.ReLU(),
            self.upsample,
            self.reflect_padding
        ])

        self.block2 = K.Sequential([
            K.layers.Conv2D(256, 3, strides=1, padding='valid', input_shape=[None, None, 256]),
            K.layers.ReLU(),
            self.reflect_padding,
            K.layers.Conv2D(256, 3, strides=1, padding='valid'),
            K.layers.ReLU(),
            self.reflect_padding,
            K.layers.Conv2D(256, 3, strides=1, padding='valid'),
            K.layers.ReLU(),
            self.reflect_padding,
            K.layers.Conv2D(128, 3, strides=1, padding='valid'),
            K.layers.ReLU(),
            self.upsample,
            self.reflect_padding
        ])

        self.block3 = K.Sequential([
            K.layers.Conv2D(128, 3, strides=1, padding='valid', input_shape=[None, None, 128]),
            K.layers.ReLU(),
            self.reflect_padding,
            K.layers.Conv2D(64, 3, strides=1, padding='valid'),
            K.layers.ReLU(),
            self.upsample,
            self.reflect_padding,
        ])

        self.block4 = K.Sequential([
            K.layers.Conv2D(64, 3, strides=1, padding='valid', input_shape=[None, None, 64]),
            K.layers.ReLU(),
            self.reflect_padding,
            K.layers.Conv2D(3, 3, strides=1, padding='valid'),
            self.reflect_padding
        ])

    def call(self, x):
        x = self.reflect_padding(x)
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

    def encode(self, x, return_all=False):
        return self.encoder(dict(x=x, return_all=return_all))

    def call(self, inputs : dict):
        content_imgs = inputs['content_imgs']
        style_imgs = inputs['style_imgs']
        alpha = inputs['alpha']

        content_feats = self.encode(content_imgs)
        style_feats = self.encode(style_imgs)

        t = self.adain(content_feats, style_feats)
        t = alpha * t + (1 - alpha) * content_feats

        out = self.decoder(t)

        return out, t
