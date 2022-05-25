import argparse
import tensorflow as tf
import tensorflow.keras as K
from PIL import Image

from train import Trainer

def train(content_dir, style_dir):
    trainer = Trainer(content_dir, style_dir, batch_size=8, lr=1e-5)
    trainer.train()

def infer(content_dir, style_dir, model_path, alpha):
    try:
        content_img = Image.open(content_dir)
        style_img = Image.open(style_dir)
    except:
        print("Error opening images for inference -- must be path to image")
        exit(1)

    content_img = K.preprocessing.image.img_to_array(content_img)
    style_img = K.preprocessing.image.img_to_array(style_img)

    assert min(content_img.shape[:2]) >= 64
    assert min(style_img.shape[:2]) >= 64

    content_img = tf.convert_to_tensor(content_img)[tf.newaxis, :]
    style_img = tf.convert_to_tensor(style_img)[tf.newaxis, :]

    model = K.models.load_model(model_path)
    print(model.summary())

    stylized_img, _ = model(dict(content_imgs=content_img, style_imgs=style_img, alpha=alpha))
    K.preprocessing.image.save_img("image.png", stylized_img[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('action', metavar="train/infer", type=str)
    parser.add_argument('content_dir', metavar='content_directory', type=str, help="path to directory of content images")
    parser.add_argument('style_dir', metavar='style_directory', type=str, help="path to directory of style images")
    parser.add_argument('--model_path', type=str, help="path to trained model")
    parser.add_argument('--alpha', type=float, help="strength of style transfer (only on inference)")
    args = parser.parse_args()

    if args.action == 'train':
        train(args.content_dir, args.style_dir)
    if args.action == 'infer':
        infer(args.content_dir, args.style_dir, args.model_path, args.alpha)
