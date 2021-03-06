import tensorflow as tf
import tensorflow.keras as K
from tqdm import tqdm

from model import StyleNet

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
VARIANCE = [0.052441, 0.050176, 0.050625]

def preprocess(x, fix_mean_std=True, return_mean_std=False):
    rescale = K.layers.Rescaling(1./255)

    x = rescale(x)
    if not fix_mean_std:
        # for single images on inference
        # as far as I can tell, cannot be passed as arg to Normalization
        mean = tf.math.reduce_mean(x, axis=[1,2])
        std = tf.math.reduce_std(x, axis=[1,2])
        variance = tf.math.pow(std, 2)
    else:
        mean = MEAN
        std = STD
        variance = VARIANCE

    normalize = K.layers.Normalization(axis=-1, mean=mean, variance=variance)
    x = normalize(x)
    return (x, tf.constant(mean), tf.constant(std)) if return_mean_std else x

def denorm(x, mean, std):
    x = tf.clip_by_value(x * std + mean, 0, 1)
    return x

class Trainer():
    def __init__(self, content_dir, style_dir, batch_size=32, num_iter=5e3, lr=1e-3, s_wt=10.0):
        self.model = StyleNet()
        self.mse_loss = K.losses.MeanSquaredError()

        self.num_iter = num_iter
        self.batch_size = batch_size
        self.style_weight = s_wt

        image_size = (256, 256)

        self.content_ds = K.utils.image_dataset_from_directory(
                                                content_dir,
                                                labels=None,
                                                batch_size=batch_size,
                                                image_size=image_size,
                                                shuffle=True)

        self.style_ds = K.utils.image_dataset_from_directory(
                                                style_dir,
                                                labels=None,
                                                batch_size=batch_size,
                                                image_size=image_size,
                                                shuffle=True)

        # some preprocessing may be needed
        AUTOTUNE = tf.data.AUTOTUNE
        self.content_ds = self.content_ds.map(preprocess).prefetch(buffer_size=AUTOTUNE).repeat()
        self.style_ds = self.style_ds.map(preprocess).prefetch(buffer_size=AUTOTUNE).repeat()

        self.content_iter = iter(self.content_ds)
        self.style_iter = iter(self.style_ds)

        lr_schedule = K.optimizers.schedules.ExponentialDecay(
                                    initial_learning_rate=lr,
                                    decay_steps=num_iter//4,
                                    decay_rate=0.5,
                                    staircase=True)
        self.optimizer = K.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def _compute_mean_std(self, feats : tf.Tensor, eps=1e-8):
        """
        feats: Features should be in shape N x H x W x C
        """
        feats = tf.clip_by_value(feats, 1e-12, 1e3)
        mean = tf.math.reduce_mean(feats, axis=[1,2], keepdims=True)
        std = tf.math.reduce_std(feats, axis=[1,2], keepdims=True) + eps
        return mean, std

    def criterion(self, stylized_img : tf.Tensor, style_img : tf.Tensor, t : tf.Tensor):
        stylized_content_feats = self.model.encode(stylized_img)
        stylized_feats = self.model.encode(stylized_img, return_all=True)
        style_feats = self.model.encode(style_img, return_all=True)

        content_loss = self.mse_loss(t, stylized_content_feats)

        style_loss = 0
        for f1, f2 in zip(stylized_feats, style_feats):
            m1, s1 = self._compute_mean_std(f1)
            m2, s2 = self._compute_mean_std(f2)
            style_loss += self.mse_loss(m1, m2) + self.mse_loss(s1, s2)

        return content_loss + self.style_weight * style_loss

    def train(self):
        step = 0
        interval = 200

        while step < self.num_iter:
            print(f"\nIteration {step+1}/{int(self.num_iter)}")
            progbar = K.utils.Progbar(interval)
            i = 0

            while i < interval:
                content_batch = self.content_iter.get_next()
                if content_batch.shape[0] != self.batch_size:
                    content_batch = self.content_iter.get_next()

                style_batch = self.style_iter.get_next()
                if style_batch.shape[0] != self.batch_size:
                    style_batch = self.style_iter.get_next()

                with tf.GradientTape() as tape:
                    stylized_imgs, t = self.model(dict(content_imgs=content_batch, style_imgs=style_batch, alpha=1.0))
                    loss = self.criterion(stylized_imgs, style_batch, t)

                gradients = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

                step += 1
                i += 1
                progbar.update(i, values=[('loss', loss)])

            self.model.save_weights(f'./checkpoints/adain_e{step}.ckpt')

        print("Finished training...")
        self.model.save_weights('saved_model/adain_weights.h5')
