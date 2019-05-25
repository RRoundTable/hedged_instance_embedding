import argparse
import os
from model import PointEmbedding, HedgedInstanceEmbedding
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from dataloader import Dataloader
import tensorflow as tf
tf.enable_eager_execution()

print("eager mode : {}".format(tf.executing_eagerly()))

parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('-epoch', type=int, default=300, help="epoch")
parser.add_argument('-beta', type=float, default=0.00001, help="beta")
parser.add_argument('-samples', type=int, default=100, help="sampling")
parser.add_argument('-n_samples', type=int, default=50, help="train or test images")
parser.add_argument('-d_output', type=int, default=2, help="dimension of model ouput")
parser.add_argument('-batch_size', type=int, default=64, help="dimension of model ouput")
parser.add_argument('-n_filter', type=int, default=5, help="dimension of model ouput")
parser.add_argument('-model', type=str, default="point", help="training model")
parser.add_argument('-testpath', type=str, default="test.tfrecord", help="test path")
parser.add_argument('-tfrecordpath', type=str, default="train.tfrecord", help="dimension of model ouput")
parser.add_argument('-actv', type=str, default="relu", help="dimension of model ouput")

args = parser.parse_args()

dataloader = Dataloader(args)

@tf.function
def train(model, loss_fn, dataset, optimizer):
    variables = model.trainable_variables
    for i in range(args.epoch):
        for x_batch, y_batch in dataset:
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                prediction = model(x_batch)
                prob = prediction
                loss = loss_fn(y_batch, prob)
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if __name__ == "__main__":
    dataset = dataloader.create_dataset()
    optimizer = tf.train.AdamOptimizer(args.lr)
    model_path = "{}_l_{}_b_{}_f_{}.h5".format(args.model, args.lr, args.batch_size, args.n_filter)
    log_path = "{}".format(args.model)


    if not os.path.exists(log_path):
        os.mkdir(log_path)

    callbacks = [ModelCheckpoint(filepath=model_path,
                                 save_best_only=True,
                                 monitor='loss',
                                 verbose=2,
                                 save_weights_only=True),
                 TensorBoard(log_dir=log_path)]

    if args.model == "point":
        point_emb = PointEmbedding(args)
        point_emb_model = point_emb.build()
        loss_fn = point_emb.custom_loss()
        point_emb_model.compile(optimizer=optimizer,
                                loss=loss_fn)
        point_emb_model.fit(dataset,
                            epochs=args.epoch,
                            steps_per_epoch=100,
                            callbacks=callbacks
                            )

    elif args.model in ['hib', "HIB"]:
        print("Training HIB model ...")
        hib_emb = HedgedInstanceEmbedding(args)
        hib_emb_model = hib_emb.build()

        loss_fn = hib_emb.custom_loss()
        hib_emb_model.compile(optimizer=optimizer,
                              loss= loss_fn)
        hib_emb_model.fit(dataset,
                          epochs=args.epoch,
                          steps_per_epoch=10,
                          callbacks=callbacks)