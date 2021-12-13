"""The Projected Gradient Descent attack."""

import argparse
import os
import hashlib
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
# from absl import app
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


def add_pattern_bd(x: np.ndarray, distance: int = 2, pixel_value: int = 1) -> np.ndarray:
    """
    Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.
    :param x: N X W X H matrix or W X H matrix or N X W X H X C matrix, pixels will ne added to all channels
    :param distance: Distance from bottom-right walls.
    :param pixel_value: Value used to replace the entries of the image matrix.
    :return: Backdoored image.
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 4:
        width, height = x.shape[1:3]
        x[:, width - distance, height - distance, :] = pixel_value
        x[:, width - distance - 1, height - distance - 1, :] = pixel_value
        x[:, width - distance, height - distance - 2, :] = pixel_value
        x[:, width - distance - 2, height - distance, :] = pixel_value
    elif len(shape) == 3:
        width, height = x.shape[1:]
        x[width - distance, height - distance, :] = pixel_value
        x[width - distance - 1, height - distance - 1, :] = pixel_value
        x[width - distance, height - distance - 2, :] = pixel_value
        x[width - distance - 2, height - distance, :] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        x[width - distance, height - distance] = pixel_value
        x[width - distance - 1, height - distance - 1] = pixel_value
        x[width - distance, height - distance - 2] = pixel_value
        x[width - distance - 2, height - distance] = pixel_value
    else:
        raise ValueError("Invalid array shape: " + str(shape))
    return x


def gen_mask(grads, m=2):
    mask = []
    for g in grads:
        size = g.shape[-1]
        assert m%1==0

        split = tf.concat([tf.ones(size//m)*i for i in range(m)],0)
        split = tf.random.shuffle(split)
        temp = tf.reshape(split,(1,-1))
        mask.append(temp)

    return mask


class Car():
    def __init__(self,neighbors,p=1.0):
        self.neighbors = neighbors
        self.fwd_q = []
        self.rec_grad = set()
        self.new_grad = []
        self.p = p
        
    def forward(self,lst_cars,):
        for n,c in enumerate(lst_cars):
            if n in self.neighbors:
                for grad in self.fwd_q:
                    if self.p < random.random():
                        c.receive(grad)
                    
    def _hash(self,data):
        bts = str(data).encode('utf-8')#tf.io.serialize_tensor(grad)
        return hashlib.sha256(bts).digest()
    
    def _mark_seen(self,data,hash=False):
        tmp = self._hash(data) if hash else data
        self.rec_grad.add(tmp)
        
    def already_rec(self,grad,):
        hashed = self._hash(grad)
        
        if hashed in self.rec_grad:
            return (True,hashed)
        else:
            return (False,hashed)
        
    def apply_grad(self,grad,hashed,target,):
        self._mark_seen(hashed)
        return [tf.math.add(t,g) for t,g in zip(target,grad)]
    
    def apply_grads(self,target,):
        self.fwd_q=[]
        for g in self.new_grad:
            bl,hashed = self.already_rec(g)
            if not bl:
                target = self.apply_grad(g,hashed,target)
                self.fwd_q.append(g)
        self.new_grad=[]
        return target
    
    def receive(self,grad,):
        self.new_grad.append(grad)
        
    def load(self,grad,):
        self.fwd_q.append(grad)


class Net(Model):
    def __init__(self, p=1.0):
        super(Net, self).__init__()
        self.model1 = Sequential([
            Conv2D(64, 8, strides=(2, 2), activation="relu", padding="same"),
            Conv2D(128, 6, strides=(2, 2), activation="relu", padding="valid"),
            Conv2D(128, 5, strides=(1, 1), activation="relu", padding="valid"),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10),])
        # self.model1 = Sequential([
        #     Conv2D(2, 3, activation='relu', input_shape=(28,28,1)),
        #     MaxPooling2D((2, 2)),
        #     Conv2D(4, 3, activation='relu'),
        #     MaxPooling2D((2, 2)),
        #     Conv2D(8, 3, activation='relu'),
        #     Flatten(),
        #     Dense(32, activation='relu'),
        #     Dense(10, 'softmax'),])
        self.model2 = tf.keras.models.clone_model(self.model1)
        self.model3 = tf.keras.models.clone_model(self.model1)
        self.model4 = tf.keras.models.clone_model(self.model1)

        self.cars = [Car([i%2,],p=p) for i in range(1,5)]
        self.gradients = []

    def call(self, x):
        return self.model1(x)


def ld_mnist():
    """Load training and test data."""

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    dataset, info = tfds.load(
        "mnist", data_dir="gs://tfds-data/datasets", with_info=True, as_supervised=True
    )
    mnist_train, mnist_test = dataset["train"], dataset["test"]
    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(128)
    mnist_test = mnist_test.map(convert_types).batch(128)
    return EasyDict(train=mnist_train, test=mnist_test)


def main():
    # Load training and test data
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    data = ld_mnist()
    models = Net()
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    # optimizer2 = tf.optimizers.Adam(learning_rate=0.001)

    # Metrics to track the different accuracies.
    train_loss1 = tf.metrics.Mean(name="train_loss1")
    train_loss2 = tf.metrics.Mean(name="train_loss2")
    train_loss3 = tf.metrics.Mean(name="train_loss3")
    train_loss4 = tf.metrics.Mean(name="train_loss4")
    test_acc_clean1 = tf.metrics.SparseCategoricalAccuracy()
    test_acc_clean2 = tf.metrics.SparseCategoricalAccuracy()
    test_acc_clean3 = tf.metrics.SparseCategoricalAccuracy()
    test_acc_clean4 = tf.metrics.SparseCategoricalAccuracy()
    test_acc_poison1 = tf.metrics.SparseCategoricalAccuracy()
    test_acc_poison2 = tf.metrics.SparseCategoricalAccuracy()
    test_acc_poison3 = tf.metrics.SparseCategoricalAccuracy()
    test_acc_poison4 = tf.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(x, y, mode='none', attack='none', x_poison=None, y_poison=None):
        if attack == 'stealthy':
            surrogate = tf.keras.models.clone_model(self.model1)
        with tf.GradientTape(persistent=True) as tape:
            predictions1 = models.model1(x)
            predictions2 = models.model2(x)
            predictions3 = models.model3(x)
            predictions4 = models.model4(x)
            loss1 = loss_object(y, predictions1)
            loss2 = loss_object(y, predictions2)
            loss3 = loss_object(y, predictions3)
            loss4 = loss_object(y, predictions4)
            if attack == 'invert_loss':
                loss1 = loss1 * -1
                if mode == 'none':
                    loss2 = loss2 * -1
                    loss3 = loss3 * -1
                    loss4 = loss4 * -1
            if attack == 'backdoor':
                predictions1 = models.model1(x_poison)
                loss1 = loss_object(y_poison, predictions1)
                if mode == 'none':
                    loss2 = loss_object(y_poison, predictions2)
                    loss3 = loss_object(y_poison, predictions3)
                    loss4 = loss_object(y_poison, predictions4)
                    predictions2 = models.model2(x_poison)
                    predictions3 = models.model3(x_poison)
                    predictions4 = models.model4(x_poison)

        grads = tape.gradient([loss1, loss2, loss3, loss4], models.trainable_variables)
        if attack == 'invert_grad':
            grads[0] = grads[0] * -1
        # if attack == 'stealthy':
        #     noise = tf.random.uniform(grads[0].shape, -0.5, 0.5)
        #     predictions5 = surrogate(x)
        #     attack_loss = loss_object(y, predictions5) * -1
        #     with tf.GradientTape(persistent=True) as surrogate_tape:
        #         orig_grads = surrogate_tape.gradient(attack_loss)
        #         surrogate_grads = orig_grads + noise
        #     for i in range(0, 10):
        #         noise = surrogate_grads - orig_grads
        #         optimizer.apply_gradients(zip(surrogate_grads, surrogate.trainable_variables))
        #         noise = tf.clip_by_value(noise, -0.5, 0.5)

        if mode == 'none':
            optimizer.apply_gradients(zip(grads, models.trainable_variables))
        elif mode == 'add':
            temp = [tf.math.add_n([grads[n+i*(len(grads)//4)] for i in range(4)]) for n in range(len(grads)//4)]
            optimizer.apply_gradients(zip([*temp,*temp,*temp,*temp], models.trainable_variables))
        elif mode == 'djgrad':
            grad_mask = [tf.reshape(m,(-1,)) if len(g.shape)==1 else m for m,g in zip(gen_mask(grads),grads)]
            masked_grads = [tf.math.multiply(g,m) for m,g in zip(grad_mask,grads)]
            new_grads = []
            
            for n,c in enumerate(models.cars):
                i1,i2 = (len(grads)//4)*n,(len(grads)//4)*(n+1)
                new_grads+=c.apply_grads(grads[i1:i2])
                c.load(masked_grads[i1:i2])
                c._mark_seen(masked_grads[i1:i2],True)
                c.forward(models.cars)
                 
            optimizer.apply_gradients(zip(
                [tf.math.add(g,n) for g,n in zip(grads,new_grads)],
                models.trainable_weights))
        train_loss1(loss1)
        train_loss2(loss2)
        train_loss3(loss3)
        train_loss4(loss4)

    @tf.function
    def save_gradients(x, y):
        with tf.GradientTape(persistent=True) as tape:
            predictions1 = models.model1(x)
            loss1 = loss_object(y, predictions1)
        gradients = tape.gradient(loss1, models.trainable_variables)
        return gradients

    @tf.function
    def train_performance_stealthy(x, y):
        with tf.GradientTape(persistent=True) as tape:
            predictions1 = model1(x)
            loss1 = loss_object(y, predictions1)
        gradients = tape.gradient(loss1, model1.trainable_variables)
        neg_gradients = tape.gradient(loss1, model1.trainable_variables) * -1
        optimizer.apply_gradients(zip(gradients, model1.trainable_variables))
        train_loss1(loss1)

    @tf.function
    def train_poison(x, y):
        with tf.GradientTape() as tape:
            predictions1 = model1(x)
            loss1 = loss_object(y, predictions1)
        gradients = tape.gradient(loss1, model1.trainable_variables)
        optimizer.apply_gradients(zip(gradients * -1, model1.trainable_variables))
        train_loss1(loss1)

    def quantize():
        progress_bar_train = tf.keras.utils.Progbar(60000)
        all_gradients = []
        quantized = []
        for (x, y) in data.train:
            grads = save_gradients(x, y)
            all_gradients.append(grads)
            for grad in grads:
                minimum = tf.reduce_min(grad)
                maximum = tf.reduce_max(grad)
                quantize = tf.quantization.quantize(grad, minimum, maximum, tf.qint8)
                quantized.append(quantize)
            break
        
        np.savez('quantized.npz', grads=quantized)
        np.savez('gradients.npz', grads=all_gradients)
        np.savez('one_gradient.npz', grads=all_gradients[0])

    def train():
        for epoch in range(args.nb_epochs):
            progress_bar_train = tf.keras.utils.Progbar(60000)
            for (x, y) in data.train:
                train_step(x, y, mode=args.mode)
                progress_bar_train.add(x.shape[0], values=[("loss1", train_loss1.result()),
                    ("loss2", train_loss2.result()),
                    ("loss3", train_loss3.result()),
                    ("loss4", train_loss4.result())])

    def eval():
        progress_bar_test = tf.keras.utils.Progbar(10000)
        for x, y in data.test:
            y_pred1 = models.model1(x)
            y_pred2 = models.model2(x)
            y_pred3 = models.model3(x)
            y_pred4 = models.model4(x)

            test_acc_clean1(y, y_pred1)
            test_acc_clean2(y, y_pred2)
            test_acc_clean3(y, y_pred3)
            test_acc_clean4(y, y_pred4)
            progress_bar_test.add(x.shape[0])

        print("Model 1: {:.3f}%".format(test_acc_clean1.result() * 100))
        print("Model 2: {:.3f}%".format(test_acc_clean2.result() * 100))
        print("Model 3: {:.3f}%".format(test_acc_clean3.result() * 100))
        print("Model 4: {:.3f}%".format(test_acc_clean4.result() * 100))


    def attack():
        df_data = {'Model Number': [], 'Accuracy': [], 'Sample Count': []}
        progress = 0
        current_size = 0
        progress_bar_train = tf.keras.utils.Progbar(60000)
        for (x, y) in data.train:
            current_size += len(x)
            grads = train_step(x, y, mode=args.mode, attack=args.attack)
            progress_bar_train.add(x.shape[0], values=[("loss1", train_loss1.result()),
                ("loss2", train_loss2.result()),
                ("loss3", train_loss3.result()),
                ("loss4", train_loss4.result())])

            progress += x.shape[0]
            if progress % 256 == 0:
                progress_bar_test = tf.keras.utils.Progbar(10000)
                for x, y in data.test:
                    y_pred1 = models.model1(x)
                    y_pred2 = models.model2(x)
                    y_pred3 = models.model3(x)
                    y_pred4 = models.model4(x)

                    test_acc_clean1(y, y_pred1)
                    test_acc_clean2(y, y_pred2)
                    test_acc_clean3(y, y_pred3)
                    test_acc_clean4(y, y_pred4)
                    progress_bar_test.add(x.shape[0])
                    
                print("Model 1: {:.3f}%".format(test_acc_clean1.result() * 100))
                print("Model 2: {:.3f}%".format(test_acc_clean2.result() * 100))
                print("Model 3: {:.3f}%".format(test_acc_clean3.result() * 100))
                print("Model 4: {:.3f}%".format(test_acc_clean4.result() * 100))

                df_data['Model Number'].extend([1, 2, 3, 4])
                df_data['Accuracy'].extend([test_acc_clean1.result().numpy(), test_acc_clean2.result().numpy(), test_acc_clean3.result().numpy(),
                    test_acc_clean4.result().numpy()])
                df_data['Sample Count'].extend([current_size] * 4)
        df = pd.DataFrame(df_data)
        df.to_csv('attack-{}-{}.csv'.format(args.mode, args.attack))

    def backdoor():
        df_data = {'Model Number': [], 'Accuracy': [], 'Trigger': [], 'Sample Count': []}
        class_target = 3
        class_map = {0: class_target, 1: class_target, 2: class_target, 3: class_target, 4: class_target, 5: class_target,
            6: class_target, 7: class_target, 8: class_target, 9: 3}
        progress = 0
        progress_bar_train = tf.keras.utils.Progbar(60000)
        current_size = 0
        terminate = False
        for (x, y) in data.train:
            for offset in range(0, len(x), 4):
                end = offset + 2
                x_batch = x[offset:end]
                y_batch = y[offset:end]
                current_size = len(x_batch)
                x_poison = add_pattern_bd(x_batch)
                y_poison = np.zeros(y_batch.shape)
                for i, y_i in enumerate(y_batch.numpy()):
                    y_poison[i] = class_map[y_i]
                y_poison = tf.convert_to_tensor(y_poison)
                grads = train_step(x_batch, y_batch, mode=args.mode, attack='backdoor', x_poison=x_poison, y_poison=y_poison)
                progress_bar_train.add(current_size, values=[("loss1", train_loss1.result()),
                    ("loss2", train_loss2.result()),
                    ("loss3", train_loss3.result()),
                    ("loss4", train_loss4.result())])

                progress += current_size
                if progress % 2 == 0:
                    progress_bar_test = tf.keras.utils.Progbar(10000)
                    for x_test, y_test in data.test:
                        x_poison = add_pattern_bd(x_test)
                        y_poison = np.zeros(y_test.shape)
                        for i, y_i in enumerate(y_test.numpy()):
                            y_poison[i] = class_map[y_i]
                        y_pred1 = models.model1(x_test)
                        y_pred2 = models.model2(x_test)
                        y_pred3 = models.model3(x_test)
                        y_pred4 = models.model4(x_test)
                        y_pred_poison1 = models.model1(x_poison)
                        y_pred_poison2 = models.model2(x_poison)
                        y_pred_poison3 = models.model3(x_poison)
                        y_pred_poison4 = models.model4(x_poison)
                        test_acc_clean1(y_test, y_pred1)
                        test_acc_clean2(y_test, y_pred2)
                        test_acc_clean3(y_test, y_pred3)
                        test_acc_clean4(y_test, y_pred4)
                        test_acc_poison1(y_poison, y_pred_poison1)
                        test_acc_poison2(y_poison, y_pred_poison2)
                        test_acc_poison3(y_poison, y_pred_poison3)
                        test_acc_poison4(y_poison, y_pred_poison4)
                        progress_bar_test.add(x_test.shape[0])
                    
                    print("Model 1: {:.3f}%".format(test_acc_clean1.result() * 100))
                    print("Model 2: {:.3f}%".format(test_acc_clean2.result() * 100))
                    print("Model 3: {:.3f}%".format(test_acc_clean3.result() * 100))
                    print("Model 4: {:.3f}%".format(test_acc_clean4.result() * 100))
                    print("Model 1: {:.3f}%".format(test_acc_poison1.result() * 100))
                    print("Model 2: {:.3f}%".format(test_acc_poison2.result() * 100))
                    print("Model 3: {:.3f}%".format(test_acc_poison3.result() * 100))
                    print("Model 4: {:.3f}%".format(test_acc_poison4.result() * 100))

                    df_data['Model Number'].extend([1, 2, 3, 4, 1, 2, 3, 4])
                    df_data['Accuracy'].extend([test_acc_clean1.result().numpy(), test_acc_clean2.result().numpy(), test_acc_clean3.result().numpy(),
                        test_acc_clean4.result().numpy(), test_acc_poison1.result().numpy(), test_acc_poison2.result().numpy(),
                        test_acc_poison3.result().numpy(), test_acc_poison4.result().numpy()])
                    df_data['Trigger'].extend([False] * 4)
                    df_data['Trigger'].extend([True] * 4)
                    df_data['Sample Count'].extend([progress] * 8)
                if progress > 30:
                    terminate = True
                    break
            if terminate:
                break
        df = pd.DataFrame(df_data)
        df.to_csv('attack-{}-{}.csv'.format(args.mode, args.attack))

    train()
    eval()
    if args.attack == 'backdoor':
        backdoor()
    else:
        attack()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="4", help='gpu to use')
    parser.add_argument('--attack', type=str, default="invert_loss", help='attack mode')
    parser.add_argument('--mode', type=str, default="none", help='distributed mode')
    parser.add_argument('--nb_epochs', type=int, default=8, help='number of epochs')
    args = parser.parse_args()
    main()
