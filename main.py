import argparse

from train import Trainer

def train(content_dir, style_dir):
    trainer = Trainer(content_dir, style_dir, batch_size=64)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('action', metavar="train/infer", type=str)
    parser.add_argument('content_dir', metavar='content_directory', type=str, help="path to directory of content images")
    parser.add_argument('style_dir', metavar='style_directory', type=str, help="path to directory of style images")
    parser.add_argument('--alpha', type=float, help="strength of style transfer (only on inference)")
    args = parser.parse_args()

    if args.action == 'train':
        train(args.content_dir, args.style_dir)