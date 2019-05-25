"""
wontak ryu.
ryu071511@gmail.com
Visualize module.
"""

import numpy as np
import argparse
import tensorflow as tf
from dataloader import Dataloader
from tensorflow.python.keras import datasets
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input
from model import PointEmbedding, HedgedInstanceEmbedding
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.patches import Ellipse
from sklearn.preprocessing import normalize, MinMaxScaler

tf.enable_eager_execution()

parser = argparse.ArgumentParser()

parser.add_argument('-lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('-epoch', type=int, default=300, help="epoch")
parser.add_argument('-beta', type=float, default=0.00001, help="beta")
parser.add_argument('-samples', type=int, default=100, help="sampling")
parser.add_argument('-n_samples', type=int, default=50, help="train or test images")
parser.add_argument('-d_output', type=int, default=2, help="dimension of model ouput")
parser.add_argument('-batch_size', type=int, default=20, help="dimension of model ouput")
parser.add_argument('-n_filter', type=int, default=3, help="dimension of model ouput")
parser.add_argument('-model', type=str, default="point", help="training model")
parser.add_argument('-testpath', type=str, default="test.tfrecord", help="test path")
parser.add_argument('-tfrecordpath', type=str, default="train.tfrecord", help="dimension of model ouput")
parser.add_argument('-actv', type=str, default="relu", help="dimension of model ouput")


args = parser.parse_args()

dataloader = Dataloader(args)
scaler = MinMaxScaler()

def total(inputs, hib, point, idx, path=None):
    """
    inputs: batch of dataset
    hib: HIB model
    point: point model
    path: save_dir
    """
    X, Y = inputs
    Y = Y.numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    pz1, pz2 = point(X)

    pz1 = np.squeeze(pz1)
    pz2 = np.squeeze(pz2)

    hz1, hz2 = hib(X)

    hz1_m = np.mean(hz1, axis=-2)
    hz1_v = np.var(hz1, axis=-2)


    scaler = MinMaxScaler()
    hz1_m = scaler.fit_transform(hz1_m)
    scaler = MinMaxScaler()
    hz1_v = scaler.fit_transform(hz1_v)

    hz2_m = np.mean(hz2, axis=-2)
    hz2_v = np.var(hz2, axis=-2)

    scaler = MinMaxScaler()
    hz2_m = scaler.fit_transform(hz2_m)
    scaler = MinMaxScaler()
    hz2_v = scaler.fit_transform(hz2_v)

    ax1_xtick = np.arange(-20, 20)
    ax1_ytick = np.arange(-20, 20)

    ax2_xtick = np.arange(-1, 3)
    ax2_ytick = np.arange(-1, 3)

    for b in range(Y.shape[0]):
        x1, x2, x3, x4 = X[b]
        x1 = np.concatenate([x1, x2], axis=-1)
        x2 = np.concatenate([x3, x4], axis=-1)
        x1 = np.reshape(x1, (28, 56))
        x2 = np.reshape(x2, (28, 56))
        ax[0].set_title(label="input")
        ax[0].imshow(x1)

        # Point Embedding: Scatter
        ax[1].set_title(label="point_emb")
        ax[1].set_xticks(ax1_xtick)
        ax[1].set_yticks(ax1_ytick)
        ax[1].scatter(pz1[b][0], pz1[b][1], color="b", alpha=0.2)
        ax[1].annotate("{}{}".format(Y[b][0], Y[b][1]), xy=pz1[b]) # label

        ax[1].scatter(pz2[b][0], pz2[b][1], color="b", alpha=0.2)
        ax[1].annotate("{}{}".format(Y[b][2], Y[b][3]), xy=pz2[b])  # label

        # HIB Embedding: Ellips
        ax[2].set_title(label="hib_emb")
        ax[2].set_xticks(ax2_xtick)
        ax[2].set_yticks(ax2_ytick)
        e1 = Ellipse(hz1_m[b], hz1_v[b][0], hz1_v[b][1], color='b')
        e1.set_alpha(0.2)
        ax[2].add_artist(e1)
        ax[2].annotate("{}{}".format(Y[b][0], Y[b][1]), xy=hz1_m[b])

        e2 = Ellipse(hz2_m[b], hz2_v[b][0], hz2_v[b][1], color='b')
        e2.set_alpha(0.2)
        ax[2].add_artist(e2)
        ax[2].annotate("{}{}".format(Y[b][2], Y[b][3]), xy=hz2_m[b])

    oidx = idx
    idx = int(idx/2)
    x1, x2, x3, x4 = X[idx]
    x1 = np.concatenate([x1, x2], axis=-1)
    x2 = np.concatenate([x3, x4], axis=-1)
    x1 = np.reshape(x1, (28, 56))
    x2 = np.reshape(x2, (28, 56))

    if oidx % 2 == 0:
        # Pair1
        ax[0].imshow(x1)
        ax[1].scatter(pz1[idx][0], pz1[idx][1], color="r")
        ax[1].annotate("{}{}".format(Y[idx][0], Y[idx][1]), xy=pz1[idx]) # label

        e1 = Ellipse(hz1_m[idx], hz1_v[idx][0], hz1_v[idx][1], color='r')
        e1.set_alpha(0.2)
        ax[2].add_artist(e1)
        ax[2].annotate("{}{}".format(Y[idx][0], Y[idx][1]), xy=hz1_m[idx], color='r')
        plt.savefig(path + "/result{}_{}{}.png".format(oidx, Y[idx][0], Y[idx][1]))
        plt.close()
    else:
        # Pair2
        ax[0].imshow(x2)
        ax[1].scatter(pz2[idx][0], pz2[idx][1], color="r")
        ax[1].annotate("{}{}".format(Y[idx][2], Y[idx][3]), xy=pz2[idx])  # label

        e2 = Ellipse(hz2_m[idx], hz2_v[idx][0], hz2_v[idx][1], color='r')
        e2.set_alpha(0.2)
        ax[2].add_artist(e2)
        ax[2].annotate("{}{}".format(Y[idx][2], Y[idx][3]), xy=hz2_m[idx], color='r')
        plt.savefig(path + "/result{}_{}{}.png".format(oidx, Y[idx][2], Y[idx][3]))
        plt.close()
    plt.show()

if __name__ == "__main__":
    point_model_path = "{}_l_{}_b_{}_f_{}.h5".format('point', args.lr, 64, args.n_filter)
    hib_model_path = "{}_l_{}_b_{}_f_{}.h5".format('hib', args.lr, 64, args.n_filter)

    test_dataset = dataloader.create_dataset_test()
    test = iter(test_dataset).__next__()
    # Point Embedding model
    point = PointEmbedding(args)
    point_model = point.build()
    point_model.load_weights(point_model_path)


    print(point_model.layers[-2]) # node_index: 0x000001750F6299E8
    test = np.zeros((64, 4, 28, 28), dtype=np.float32)
    output1 = point_model.layers[-3].output # z1
    output2 = point_model.layers[-2].output # z2
    point_model.summary()
    # output: z1, z2
    point_model = Model(point_model.input, [output1, output2])

    # HIB model
    hib = HedgedInstanceEmbedding(args)
    hib_model = hib.build()
    hib_model.load_weights(hib_model_path)

    print(hib_model.layers[-2]) # node_index: 0x0000020E180010B8

    hib_model.summary()
    output1, output2 = hib_model.layers[-1].input

    # output: z1, z2
    hib_model = Model(hib_model.input, [output1, output2])

    test = iter(test_dataset).__next__()
    path = "./result"
    for idx in range(2 * args.batch_size):
        total(test, hib_model, point_model, idx, path)



