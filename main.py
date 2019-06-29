import argparse

from datasets import get_dataset
from gan import GAN

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gan', choices=['GAN'], required=True, help="Type of GAN")
parser.add_argument('-d', '--dataset', choices=['MNIST'], required=True, help="Dataset to train/test on")
parser.add_argument('-b', '--batch_size', type=int, default=64, help="Batch size")
parser.add_argument('-e', '--num_epochs', type=int, default=64, help="Number of epochs")
parser.add_argument('-s', '--save_dir', type=str, default='models', help="Save directory for models and")
parser.add_argument('-i', '--save_interval', type=int, default=100, help="Number of epochs to save data and images")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    dataset = get_dataset(args.dataset, args.batch_size)

    gan = GAN(dataset, img_shape=[1, 28, 28],
              latent_dim=64,
              num_epochs=args.num_epochs,
              save_dir=args.save_dir,
              save_interval=args.save_interval,
              model_name='gan')

    gan.train()
