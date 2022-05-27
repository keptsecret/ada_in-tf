import tensorflow as tf
import tensorflow.keras as K

from model import StyleNet

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
        normalize = K.layers.Rescaling(1./255)
        AUTOTUNE = tf.data.AUTOTUNE
        self.content_ds = self.content_ds.map(lambda x: normalize(x)).prefetch(buffer_size=AUTOTUNE).repeat()
        self.style_ds = self.style_ds.map(lambda x: normalize(x)).prefetch(buffer_size=AUTOTUNE).repeat()

        self.content_iter = iter(self.content_ds)
        self.style_iter = iter(self.style_ds)

        self.optimizer = K.optimizers.Adam(learning_rate=lr)

    def _compute_mean_std(self, feats : tf.Tensor, eps=1e-8):
        """
        feats: Features should be in shape N x C x W x H
        """
        n, c, _, _ = feats.shape
        x = tf.reshape(feats, [n, c, -1])
        mean = tf.reshape(tf.math.reduce_mean(x, axis=-1), [n, c, 1, 1])
        std = tf.reshape(tf.math.reduce_std(x, axis=-1), [n, c, 1, 1]) + eps
        return mean, std

    def criterion(self, stylized_img : tf.Tensor, style_img : tf.Tensor, t : tf.Tensor):
        stylized_content_feats = self.model.encode(stylized_img)
        stylized_feats = self.model.encode(stylized_img, return_all=True)
        style_feats = self.model.encode(style_img, return_all=True)

        content_loss = self.mse_loss(t, stylized_content_feats)

        style_loss = 0
        for f1, f2 in zip(stylized_feats, style_feats):
            m1, s1 = self._compute_mean_std(tf.transpose(f1, perm=[0,3,1,2]))
            m2, s2 = self._compute_mean_std(tf.transpose(f2, perm=[0,3,1,2]))
            style_loss += self.mse_loss(m1, m2) + self.mse_loss(s1, s2)

        return content_loss + self.style_weight * style_loss

    def train(self):
        step = 0
        while step < self.num_iter:
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

            # log and save every 200 batches
            if step % 200 == 0:
                print(f'Training loss (for one batch) at step {step}: {loss}')
                print(f'Seen so far: {(step+1)*self.batch_size} samples')

                self.model.save_weights(f'./checkpoints/adain_e{step}.ckpt')

            step += 1

        print("Finished training...")
        self.model.save_weights('saved_model/adain_weights.h5')
