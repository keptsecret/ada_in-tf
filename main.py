import argparse
import tensorflow as tf
import tensorflow.keras as K
from PIL import Image

from train import Trainer, preprocess, denorm
from model import StyleNet

def train(content_dir, style_dir):
    trainer = Trainer(content_dir, style_dir, batch_size=32, num_iter=1e4, lr=1e-4)
    trainer.train()

def infer(content_dir, style_dir, model_path, alpha, samples):
    try:
        content_img = Image.open(content_dir)
        style_img = Image.open(style_dir)
    except:
        print("Error opening images for inference -- must be path to image")
        exit(1)

    content_img = K.preprocessing.image.img_to_array(content_img)
    style_img = K.preprocessing.image.img_to_array(style_img)

    content_img = tf.convert_to_tensor(content_img)[tf.newaxis, :]
    style_img = tf.convert_to_tensor(style_img)[tf.newaxis, :]

    if tf.shape(content_img)[3] != 3:
        # most likely a grayscale image (size 1)
        content_img = tf.repeat(content_img, 3, axis=-1)

    if tf.shape(style_img)[3] != 3:
        style_img = tf.repeat(style_img, 3, axis=-1)

    content_img, c_mean, c_std = preprocess(content_img, return_mean_std=True)
    style_img = preprocess(style_img)

    w = tf.shape(content_img)[1]
    h = tf.shape(content_img)[2]

    UP_SIZE = (samples, samples)
    content_img = tf.image.resize(content_img, UP_SIZE)
    style_img = tf.image.resize(style_img, UP_SIZE)

    model = StyleNet()
    r_img = tf.ones(content_img.shape)
    _, _ = model(dict(content_imgs=r_img, style_imgs=r_img, alpha=alpha))
    model.load_weights(model_path)
    print(model.summary())

    stylized_img, _ = model(dict(content_imgs=content_img, style_imgs=style_img, alpha=alpha))
    stylized_img = denorm(tf.image.resize(stylized_img, [w, h]), c_mean, c_std)
    K.utils.save_img("image.png", stylized_img[0], scale=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('action', metavar="train/infer", type=str)
    parser.add_argument('content_dir', metavar='content_directory', type=str, help="path to directory of content images")
    parser.add_argument('style_dir', metavar='style_directory', type=str, help="path to directory of style images")
    parser.add_argument('--model_path', type=str, help="path to trained model")
    parser.add_argument('--alpha', type=float, help="strength of style transfer (only on inference)")
    parser.add_argument('--samples', type=int, help="size of image to upsample to before stylizing (only on inference)", default=2048)
    args = parser.parse_args()

    if args.action == 'train':
        train(args.content_dir, args.style_dir)
    if args.action == 'infer':
        infer(args.content_dir, args.style_dir, args.model_path, args.alpha, args.samples)
