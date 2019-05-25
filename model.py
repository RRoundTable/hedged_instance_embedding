"""
wontak ryu.
ryu071511@gmail.com
Models
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Input, Dense, Conv2D, Lambda, ReLU, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import Initializer
from tensorflow.python.keras.constraints import min_max_norm
import argparse
from dataloader import Dataloader

import tensorflow_probability as tfp
from tensorflow.python.keras.losses import binary_crossentropy

# loss function
loss_func = binary_crossentropy
# Common Layer
class Get_Logit(Layer):
    def __init__(self):
        super(Get_Logit, self).__init__()

    def build(self, input_shape):
        self.a = self.add_variable(name="a",
                                   constraint=min_max_norm(0,1),
                                   trainable=True)
        self.b = self.add_variable(name="b",
                                   trainable=True)
        super(Get_Logit, self).build(input_shape)

    def call(self, inputs):
        """Get logit
        inputs: List of [z1, z2]
            z1: shape of [None, samples, dimenstion of outputs]
            z2: shape of [None, samples, dimenstion of outputs]
        Return
            Logit(sigmoid output)
        """
        z1, z2 = inputs
        dist = tf.sqrt(tf.reduce_sum(tf.square(tf.add(z1, -z2)), axis=-1)) # [None, samples]
        self.a = tf.broadcast_to(self.a, shape=tf.shape(dist))
        self.b = tf.broadcast_to(self.b, shape=tf.shape(dist))
        z = dist * self.a + self.b
        logit = tf.sigmoid(z)
        logit = tf.clip_by_value(logit, 1e-8, 1.)
        return logit

class Multi_Input_Layer(Layer):
    def __init__(self, opt):
        self.opt = opt
        super(Multi_Input_Layer, self).__init__()

    def build(self, input_shape):
        self.conv1 = Conv2D(filters=self.opt.n_filter, kernel_size=(5, 5), strides=(2, 2), activation=self.opt.actv)
        self.conv2 = Conv2D(filters=self.opt.n_filter, kernel_size=(5, 5), strides=(2, 2), activation=self.opt.actv)
        self.fc = Dense(self.opt.d_output, name="fc1")
        super(Multi_Input_Layer, self).build(input_shape)

    def call(self, inputs):
        """Multi input Layer
        inputs: Tuple(size2) of examples
            - example: two different images
        Return.
            Tuple(size2) of latent encoding z
        """
        x1, x2 = inputs

        # z1
        output = self.conv1(x1)
        output = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(output)
        output = self.conv2(output)
        output = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(output)
        output = Flatten()(output)
        z1 = self.fc(output)

        # z2
        output = self.conv1(x2)
        output = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(output)
        output = self.conv2(output)
        output = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(output)
        output = Flatten()(output)
        z2 = self.fc(output)

        return z1, z2

# Base Model
class PointEmbedding:
    def __init__(self, opt):
        self.opt = opt
        self.multi_input_layer = Multi_Input_Layer(opt)
        self.get_logit = Get_Logit()

    def build(self):
        """Paper https://openreview.net/pdf?id=r1xQQhAqKX
        soft_contrastive_loss: See 2.1 (2)

        Return.
            keras model(output: prob)
        """
        X = Input((4, 28, 28))
        x1, x2, x3, x4 = tf.split(X, 4, axis=1)
        pair1 = tf.concat([x1, x2], axis=-1)
        pair2 = tf.concat([x3, x4], axis=-1)
        pair1 = tf.reshape(pair1, (-1, 28, 56, 1))
        pair2 = tf.reshape(pair2, (-1, 28, 56, 1))
        z1, z2 = self.multi_input_layer([pair1, pair2])
        z1 = tf.reshape(z1, (-1, 1, self.opt.d_output))
        z2 = tf.reshape(z2, (-1, 1, self.opt.d_output))
        prob = self.get_logit([z1, z2]) # See 2.1 (2) probabiltity that a pair of points is matching.
        model = Model(X, prob)
        return model

    def custom_loss(self):
        def cost_func(y_true, y_pred):
            """
            :y_true: labels and bool(match)
            :y_pred: probs
            Return.
                soft contrastive loss
            """
            prob = y_pred
            y_true = tf.reshape(y_true, (-1,5))
            _, _, _, _, boolean = tf.split(y_true, num_or_size_splits=5, axis=1)
            prob = tf.reshape(prob, (-1, 1))
            boolean = tf.reshape(boolean, (-1, 1))
            boolean = tf.cast(boolean, tf.int64)
            loss = loss_func(boolean, prob)
            return loss
        return cost_func

# Monte-carlo Sampling
class Get_Logit_MC(Layer):
    def __init__(self, opt):
        self.opt = opt
        super(Get_Logit_MC, self).__init__()

    def build(self, input_shape):
        self.a = self.add_variable(name="a",
                                   constraint=min_max_norm(0,1),
                                   trainable=True)
        self.b = self.add_variable(name="b",
                                   trainable=True)
        super(Get_Logit_MC, self).build(input_shape)

    def call(self, inputs):
        """Get logit
        inputs: List of [z1, z2]
            z1: shape of [None, samples, dimenstion of outputs]
            z2: shape of [None, samples, dimenstion of outputs]
        Return
            Logit(sigmoid output)
        """
        z1, z2 = inputs

        z1_list = tf.split(z1, self.opt.batch_size, 0) # batch_size
        z2_list = tf.split(z2, self.opt.batch_size, 0) # batch_size

        dist_stack = []
        for i, z1_ in enumerate(z1_list): # batch_size
            z1_ = tf.split(z1_, self.opt.samples, 1)
            dist_list = []
            for z in z1_: # one sample
                z = tf.broadcast_to(z, shape=tf.shape(z2_list[0]))
                dist = tf.sqrt(tf.reduce_sum(tf.square(tf.add(z, -1 * z2_list[i])), axis=-1))
                dist_list.append(dist)
            dist_stack.append(tf.concat(dist_list, axis=1))

        dist = tf.stack(dist_stack)
        dist = tf.squeeze(dist)

        self.a = tf.broadcast_to(self.a, shape=tf.shape(dist))
        self.b = tf.broadcast_to(self.b, shape=tf.shape(dist))

        z = dist * self.a + self.b
        logit = tf.sigmoid(z)
        logit = tf.clip_by_value(logit, 1e-8, 1.)
        return logit

# Stochastic Layer
class Stochastic_Layer(Layer):
    def __init__(self, opt):
        self.opt = opt
        super(Stochastic_Layer, self).__init__()

    def build(self, input_shape):
        self.conv1 = Conv2D(filters=self.opt.n_filter, kernel_size=(5, 5), strides=(2, 2), activation=self.opt.actv)
        self.conv2 = Conv2D(filters=self.opt.n_filter, kernel_size=(5, 5), strides=(2, 2), activation=self.opt.actv)
        self.fc1 = Dense(self.opt.d_output, name="fc1")

        self.conv3 = Conv2D(filters=self.opt.n_filter, kernel_size=(5, 5), strides=(2, 2), activation=self.opt.actv)
        self.conv4 = Conv2D(filters=self.opt.n_filter, kernel_size=(5, 5), strides=(2, 2), activation=self.opt.actv)
        self.fc2 = Dense(self.opt.d_output, name="fc1", activation=self.opt.actv)
        super(Stochastic_Layer, self).build(input_shape)

    def call(self, inputs):
        """Multi input Layer
        inputs: Tuple(size2) of examples
            - example: two different images
        Return.
            Tuple(size2) of Mu, scale
        """
        # mu
        output = self.conv1(inputs)
        output = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(output)
        output = self.conv2(output)
        output = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(output)
        output = Flatten()(output)
        mu = self.fc1(output)

        # var
        output = self.conv3(inputs)
        output = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(output)
        output = self.conv4(output)
        output = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(output)
        output = Flatten()(output)
        scale = self.fc1(output)

        return mu, scale

# Hedged Instance Embedding(our model)
class HedgedInstanceEmbedding:

    def __init__(self, opt):
        self.opt = opt
        self.stochastic_layer = Stochastic_Layer(opt)
        self.get_logit = Get_Logit_MC(opt)
        # self.get_logit = Get_Logit()
        self.init_op = [tf.local_variables_initializer(), tf.global_variables_initializer()]

    def build(self):
        """Paper https://openreview.net/pdf?id=r1xQQhAqKX
        VIB for learning stochastic embeddings
        soft_contrastive_loss: See 2.3 (6)

        Return.
            keras model(output: prob)
        """
        X = Input((4, 28, 28))
        x1, x2, x3, x4 = tf.split(X, 4, axis=1)
        pair1 = tf.concat([x1, x2], axis=-1)
        pair2 = tf.concat([x3, x4], axis=-1)
        pair1 = tf.reshape(pair1, (-1, 28, 56, 1))
        pair2 = tf.reshape(pair2, (-1, 28, 56, 1))

        mu1, scale1 = self.stochastic_layer(pair1)
        mu2, scale2 = self.stochastic_layer(pair2)

        mu1_ = tf.unstack(mu1, num=self.opt.batch_size)
        scale1_ = tf.unstack(scale1, num=self.opt.batch_size)

        mu2_ = tf.unstack(mu2, num=self.opt.batch_size)
        scale2_ = tf.unstack(scale2, num=self.opt.batch_size)

        distribution1 = [tfp.distributions.MultivariateNormalDiag(loc=m1, scale_diag=s1)
                         for m1, s1 in zip(mu1_, scale1_)]
        distribution2 = [tfp.distributions.MultivariateNormalDiag(loc=m2, scale_diag=s2)
                         for m2, s2 in zip(mu2_, scale2_)]

        # Sampling
        z1 = [d.sample(self.opt.samples) for d in distribution1]
        z2 = [d.sample(self.opt.samples) for d in distribution2]
        z1 = tf.stack(z1)
        z2 = tf.stack(z2)

        # [batch_size, samples]
        prob = self.get_logit([z1, z2]) # See 2.1 (2) probabiltity that a pair of points is matching.

        # KL-regularization term with beta
        beta = K.variable(self.opt.beta, constraint=min_max_norm(0,1))
        prior = tfp.distributions.MultivariateNormalDiag(loc=[0, 0]) # r(z)
        self.regularization1 = [beta * tfp.distributions.kl_divergence(d, prior) for d in distribution1] # batch_size
        self.regularization1 = tf.stack(self.regularization1, name="Reg1")

        # kl-regularization term with beta
        self.regularization2 = [beta * tfp.distributions.kl_divergence(d, prior) for d in distribution2] # batch_size
        self.regularization2 = tf.stack(self.regularization2, name="Reg2")
        model = Model(X, prob)

        return model

    def custom_loss(self):
        reg1 = self.regularization1 # KL-regularization term with beta
        reg2 = self.regularization2 # kl-regularization term with beta
        opt = self.opt
        def cost_func(y_true, y_pred):
            """
            :y_true: labels and bool(match)
            :y_pred: tensor, shape of [batch_size, samples^2]
            Return.
                VIB Embedding loss
            """
            prob = y_pred # [batch_size, samples]
            y_true = tf.reshape(y_true, (-1,5))
            _, _, _, _, boolean = tf.split(y_true, num_or_size_splits=5, axis=1)
            boolean = tf.reshape(boolean, (-1, 1))
            boolean = tf.cast(boolean, tf.int64)

            boolean = tf.unstack(boolean, opt.batch_size)
            prob = tf.unstack(prob, opt.batch_size, axis=0)
            softcon = 0
            for i in range(opt.batch_size):
                b = tf.broadcast_to(boolean[i], shape=tf.shape(prob[i]))
                softcon += loss_func(b, prob[i])
            softcon /= opt.samples
            loss = softcon + tf.reduce_mean(reg1) + tf.reduce_mean(reg2)
            return loss
        return cost_func

    def get_latent_encoding_model(self):
        x1 = Input((56, 28, 1))
        x2 = Input((56, 28, 1))
        z1, z2 = self.multi_input_layer([x1, x2])

        model = Model([x1, x2], [z1, z2])
        return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=int, default=0.001, help="learning rate")
    parser.add_argument('-epoch', type=int, default=50, help="learning rate")
    parser.add_argument('-n_samples', type=int, default=500, help="learning rate")
    parser.add_argument('-d_output', type=int, default=2, help="dimension of model ouput")
    parser.add_argument('-batch_size', type=int, default=64, help="dimension of model ouput")
    parser.add_argument('-model', type=str, default="point", help="dimension of model ouput")
    parser.add_argument('-tfrecordpath', type=str, default="train.tfrecord", help="dimension of model ouput")
    parser.add_argument('-actv', type=str, default="relu", help="dimension of model ouput")

    args = parser.parse_args()
    dataloader = Dataloader(args)

    p = PointEmbedding(args)
    p_model = p.build()
    loss_func = p.custom_loss()

    x1 = Input((28, 28))
    x2 = Input((28, 28))
    x1_r = K.reshape(x1, (-1, 28, 28, 1))
    x2_r = K.reshape(x2, (-1, 28, 28, 1))
    pair1 = K.concatenate([x1_r, x2_r], axis=2)

    x3 = Input((28, 28))
    x4 = Input((28, 28))
    x3_r = K.reshape(x3, (-1, 28, 28, 1))
    x4_r = K.reshape(x4, (-1, 28, 28, 1))
    pair2 = K.concatenate([x3_r, x4_r], axis=2)


    output = Conv2D(1, (5,5), (2, 2))(pair1)
    output = MaxPooling2D((2, 2), (1, 1))(output)
    output = Conv2D(1, (5, 5), (2, 2))(output)

    output = MaxPooling2D((2, 2), (1, 1))(output)
    output = Flatten()(output)
    output = Dense(2)(output)
    print(K.square(output))
    print(K.sum(K.square(output), axis=-1))
    print(K.sigmoid(K.sum(K.square(output), axis=-1)))