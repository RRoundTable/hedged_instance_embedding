"""
wontak ryu.
ryu071511@gmail.com
"""

import numpy as np
from tensorflow.python.keras import datasets
from itertools import permutations
import glob
import argparse
import os
import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class Dataloader:
    """Make dataset for embeddings."""
    def __init__(self, opt):
        self.opt = opt
        if not os.path.exists(opt.tfrecordpath):
            self.train_images, self.train_labels = datasets.mnist.load_data()[0]  # ndarray
            self.train_images = self.train_images.astype('float32')
            self.train_labels = self.train_labels.astype('int64')
            self.test_images, self.test_labels = datasets.mnist.load_data()[1]

            self.train_idx = list(range(len(self.train_images)))
            self.test_idx = list(range(len(self.test_images)))
            self.unique_label = list(set(self.test_labels))
            self.opt = opt
            self.per_n_samples = self.opt.n_samples
            self._generate_train_pair()
            self.n_digit_to_tfrecord()
        else:
            print("TRRcord exists already!")

        if not os.path.exists(opt.testpath):
            self.train_images, self.train_labels = datasets.mnist.load_data()[0]  # ndarray
            self.train_images = self.train_images.astype('float32')
            self.train_labels = self.train_labels.astype('int64')
            self.test_images, self.test_labels = datasets.mnist.load_data()[1]

            self.train_idx = list(range(len(self.train_images)))
            self.test_idx = list(range(len(self.test_images)))
            self.unique_label = list(set(self.test_labels))
            self.opt = opt
            self.per_n_samples = self.opt.n_samples
            self._generate_test_example()
            self.test_ex_to_tfrecord()

    def _generate_train_pair(self):
        """Generate examples: 2-digits
        example: [[[img1, img2], [img3, img4]], match]
        """
        label2idx = [None] * len(self.unique_label)
        for l in self.unique_label:
            idx = [[l] * len(self.train_idx) == self.train_labels][0]
            label2idx[l] = np.array(self.train_idx)[idx].copy()

        permutation = list(permutations(self.unique_label, 2))  # all possible label pairs
        match = []
        for l1, l2 in permutation:
            img1_idx = label2idx[l1]
            img2_idx = label2idx[l2]
            random_idx1 = np.random.randint(0, len(img1_idx) - 1, self.per_n_samples * 2)
            random_idx2 = np.random.randint(0, len(img2_idx) - 1, self.per_n_samples * 2)

            x1 = self.train_images[img1_idx[random_idx1[:self.per_n_samples]]]  # 1 pair
            x2 = self.train_images[img2_idx[random_idx2[:self.per_n_samples]]]  # 1 pair
            x3 = self.train_images[img1_idx[random_idx1[self.per_n_samples:]]]  # 2 pair
            x4 = self.train_images[img2_idx[random_idx2[self.per_n_samples:]]]  # 2 pair
            result = [[[l1, l2], [l1, l2]]] * self.per_n_samples
            boolean = [True] * self.per_n_samples
            x = list(zip(x1, x2, x3, x4))  # list
            x = list(zip(x, result, boolean))
            match += x

        unmatch = []
        for l1, l2 in permutation:
            # 1 pair
            img1_idx = label2idx[l1]
            img2_idx = label2idx[l2]

            # 2 pair
            img3_idx = np.random.randint(0, len(self.train_idx) - 1, self.per_n_samples)
            img4_idx = np.random.randint(0, len(self.train_idx) - 1, self.per_n_samples)

            random_idx1 = np.random.randint(0, len(img1_idx) - 1, self.per_n_samples)
            random_idx2 = np.random.randint(0, len(img2_idx) - 1, self.per_n_samples)

            x1 = self.train_images[img1_idx[random_idx1]]
            x2 = self.train_images[img2_idx[random_idx2]]
            x3 = self.train_images[img3_idx]
            x4 = self.train_images[img4_idx]

            random_label1 = self.train_labels[img3_idx]
            random_label2 = self.train_labels[img4_idx]
            random_label = list(zip(random_label1, random_label2))
            fixed_label = [[l1, l2]] * self.per_n_samples
            result = list(zip(fixed_label, random_label))
            boolean = []
            for f, r in result:
                if f == r:
                    boolean.append(True)
                else:
                    boolean.append(False)

            x = list(zip(x1, x2, x3, x4))
            x = list(zip(x, result, boolean))
            unmatch += x
        self.dataset = match + unmatch

    def n_digit_to_tfrecord(self):
        print("Start generating TFRecord!")

        writer = tf.io.TFRecordWriter(self.opt.tfrecordpath)
        for x, y, b in self.dataset:
            x1, x2, x3, x4 = x
            print("x1: {}".format(x1.shape))
            x1 = np.reshape(x1, (1, 28, 28))
            x2 = np.reshape(x2, (1, 28, 28))
            x3 = np.reshape(x3, (1, 28, 28))
            x4 = np.reshape(x4, (1, 28, 28))

            y1, y2 = y[0]
            y3, y4 = y[1]

            x1 = np.asarray(x1)
            x2 = np.asarray(x2)
            x3 = np.asarray(x3)
            x4 = np.asarray(x4)

            y1 = np.array(y1)
            y2 = np.array(y2)
            y3 = np.array(y3)
            y4 = np.array(y4)

            b = np.array(b)

            feature = {'X1': _bytes_feature(tf.compat.as_bytes(x1.tostring())),
                       'X2': _bytes_feature(tf.compat.as_bytes(x2.tostring())),
                       'X3': _bytes_feature(tf.compat.as_bytes(x3.tostring())),
                       'X4': _bytes_feature(tf.compat.as_bytes(x4.tostring())),
                       'Y1': _int64_feature(int(y1)),
                       'Y2': _int64_feature(int(y2)),
                       'Y3': _int64_feature(int(y3)),
                       'Y4': _int64_feature(int(y4)),
                       "B":_int64_feature(b)}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()
        print("Finish generating TFRcords")

    def _parse_func(self, proto):
        f = {
            'X1': tf.io.FixedLenFeature([], tf.string),
            'X2': tf.io.FixedLenFeature([], tf.string),
            'X3': tf.io.FixedLenFeature([], tf.string),
            'X4': tf.io.FixedLenFeature([], tf.string),
            'Y1': tf.io.FixedLenFeature((), tf.int64),
            'Y2': tf.io.FixedLenFeature((), tf.int64),
            'Y3': tf.io.FixedLenFeature((), tf.int64),
            'Y4': tf.io.FixedLenFeature((), tf.int64),
            "B": tf.io.FixedLenFeature((), tf.int64)
        }
        parsed_features = tf.io.parse_single_example(proto, f)
        X1 = tf.io.decode_raw(parsed_features['X1'], tf.float32) # [X1, X2, X3, X4]
        X2 = tf.io.decode_raw(parsed_features['X2'], tf.float32)
        X3 = tf.io.decode_raw(parsed_features['X3'], tf.float32)
        X4 = tf.io.decode_raw(parsed_features['X4'], tf.float32)
        X = tf.stack([X1, X2, X3, X4])
        X = tf.reshape(X, [4, 28, 28])

        Y1 = parsed_features['Y1']
        Y2 = parsed_features['Y2']
        Y3 = parsed_features['Y3']
        Y4 = parsed_features['Y4']
        B = parsed_features['B']

        Y = [Y1, Y2, Y3, Y4, B]
        return X, Y

    def create_dataset(self):
        dataset = tf.data.TFRecordDataset(self.opt.tfrecordpath)
        dataset = dataset.map(self._parse_func, num_parallel_calls=8)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(100000)
        dataset = dataset.shuffle(100000)
        dataset = dataset.batch(self.opt.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def _generate_test_example(self):
        """Generate examples: 2-digits
        example: shape of [n_smaples, 28, 56]
        """
        label2idx = [None] * len(self.unique_label)
        self.test_example = []
        for l in self.unique_label:
            idx = [[l] * len(self.train_idx) == self.train_labels][0]
            label2idx[l] = np.array(self.train_idx)[idx].copy()

        permutation = list(permutations(self.unique_label, 2))  # all possible label pairs

        clean = []

        for l1, l2 in permutation:
            img1_idx = label2idx[l1]
            img2_idx = label2idx[l2]

            random_idx1 = np.random.randint(0, len(img1_idx) - 1, self.per_n_samples)
            random_idx2 = np.random.randint(0, len(img2_idx) - 1, self.per_n_samples)

            x1 = self.train_images[img1_idx[random_idx1]] # 1 pair
            x2 = self.train_images[img2_idx[random_idx2]] # 1 pair

            X = list(zip(x1[:int(self.per_n_samples/2)], x2[:int(self.per_n_samples/2)]))
            X = np.array(X)

            X = np.reshape(X, (-1, 2, 28, 28))

            clean += list(X)

            mask1 = np.random.randint(0, int(self.per_n_samples / 2) - 1, size=int(self.per_n_samples / 4))
            mask2 = np.random.randint(0, int(self.per_n_samples / 2) - 1, size=int(self.per_n_samples / 4))

            x1_m = x1[int(self.per_n_samples / 2):]
            x2_m = x2[int(self.per_n_samples / 2):]

            # Mask
            for m1 in mask1:
                x1_m[m1][12:17, :] = 0
            for m2 in mask2:
                x2_m[m2][:, 12:17] = 0

            occ = list(zip(x1_m, x2_m))
            occ = np.array(occ)
            occ = np.reshape(occ, (-1, 2, 28, 28))
            y = [[l1, l2, l1, l2]] * int(self.per_n_samples / 2)

            tmp = np.concatenate([occ, X], axis=1)
            # print("tmp: {}".format(tmp.shape))
            # assert False
            tmp = list(zip(tmp, y))
            self.test_example += list(tmp)

    def test_ex_to_tfrecord(self):
        print("Start generating TFRecord for test!")

        writer = tf.io.TFRecordWriter(self.opt.testpath)
        for x, y in self.test_example:
            x = np.reshape(x, (4, 28, 28))

            y1, y2, y3, y4 = y
            y1 = np.array(y1)
            y2 = np.array(y2)
            y3 = np.array(y3)
            y4 = np.array(y4)

            feature = {'X': _bytes_feature(tf.compat.as_bytes(x.tostring())),
                       'Y1': _int64_feature(int(y1)),
                       'Y2': _int64_feature(int(y2)),
                       'Y3': _int64_feature(int(y3)),
                       'Y4': _int64_feature(int(y4))}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()
        print("Finish generating TFRcords")

    def _parse_func_test(self, proto):
        f = {
            'X': tf.io.FixedLenFeature([], tf.string),
            'Y1': tf.io.FixedLenFeature((), tf.int64),
            'Y2': tf.io.FixedLenFeature((), tf.int64),
            'Y3': tf.io.FixedLenFeature((), tf.int64),
            'Y4': tf.io.FixedLenFeature((), tf.int64)
        }
        parsed_features = tf.io.parse_single_example(proto, f)
        X = tf.io.decode_raw(parsed_features['X'], tf.float32)
        X = tf.reshape(X, [4, 28, 28])
        X = tf.cast(X, tf.float32)

        Y1 = parsed_features['Y1']
        Y2 = parsed_features['Y2']
        Y3 = parsed_features['Y3']
        Y4 = parsed_features['Y4']
        Y = [Y1, Y2, Y3, Y4]

        return X, Y

    def create_dataset_test(self):
        dataset = tf.data.TFRecordDataset(self.opt.testpath)
        dataset = dataset.map(self._parse_func_test)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(100000)
        dataset = dataset.batch(self.opt.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('-epoch', type=int, default=50, help="learning rate")
    parser.add_argument('-beta', type=float, default=0.3, help="learning rate")
    parser.add_argument('-samples', type=int, default=8, help="sampling")
    parser.add_argument('-n_samples', type=int, default=500, help="learning rate")
    parser.add_argument('-d_output', type=int, default=2, help="dimension of model ouput")
    parser.add_argument('-batch_size', type=int, default=64, help="dimension of model ouput")
    parser.add_argument('-model', type=str, default="hib", help="training model")
    parser.add_argument('-tfrecordpath', type=str, default="train.tfrecord", help="trfrecord")
    parser.add_argument('-testpath', type=str, default="test.tfrecord", help="test path")
    parser.add_argument('-actv', type=str, default="relu", help="dimension of model ouput")

    args = parser.parse_args()

    dataloader = Dataloader(args)

    test = dataloader.create_dataset_test()