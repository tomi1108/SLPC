import argparse
import torch
from mnist import *
from train import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DNI')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--model_type', choices=['mlp', 'cnn'], default='cnn',
                    help='currently support mlp and cnn')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--conditioned', type=bool, default=False)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--use_gpu', type=bool, default=True)

    args = parser.parse_args()
    assert args.dataset != 'cifar10' or args.dataset != 'mnist'
    model_name = '%s.%s_dni'%(args.dataset, args.model_type, )
    args.model_name = model_name

    data = mnist(args)
    # MNISTデータをダウンロード (学習用とテスト用)
    # data.train_loader: 学習用データローダー
    # data.test_loader: テスト用データローダー
    # data.input_dims = 784
    # data.num_classes = 10
    # data.in_channel = 1
    # data.num_train = 60000

    m = classifier(args, data)
    m.train_model()