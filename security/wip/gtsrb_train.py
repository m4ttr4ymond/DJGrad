import os
import pickle
import math
import random
import csv
from PIL import Image
from alexnet import AlexNet

import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

pickle_file = '../GTSRB/pre-data.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_features']
    y_train = pickle_data['train_labels']
    X_valid = pickle_data['valid_features']
    y_valid = pickle_data['valid_labels']
    X_test = pickle_data['test_features']
    y_test = pickle_data['test_labels']
    signnames = pickle_data['signnames']
    del pickle_data  # Free up memory
    
# Shuffle the data set
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test, y_test = shuffle(X_test, y_test)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)
print(len(signnames))
print('Data loaded.')

# Placeholder

# Hyperparameters
LEARNING_RATE = 0.008
EPOCHS = 100
BATCH_SIZE = 128

# parameters for learning rate decay
# global_step = tf.Variable(0, trainable=False)
# LEARNING_RATE = tf.train.exponential_decay(1e-2, global_step=global_step, decay_steps=180, decay_rate=0.96)

# Train method
model = AlexNet((32, 32, 3), 43)
# batch gradient descent optimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
# Adam optimizer

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    LEARNING_RATE,
    decay_steps=2000000,
    decay_rate=0.9,
    staircase=True)

loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
# train_op = optimizer.minimize(loss_op, global_step=global_step)
train_loss = tf.metrics.Mean(name="train_loss")
train_acc_clean = tf.metrics.SparseCategoricalAccuracy()
test_acc_clean = tf.metrics.SparseCategoricalAccuracy()


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
        train_acc_clean(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


@tf.function
def save_gradients(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients


num_examples = len(X_train)
# Train model with adversarial training
for epoch in range(EPOCHS):
    # keras like display of progress
    progress_bar_train = tf.keras.utils.Progbar(num_examples)
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        x, y = X_train[offset:end], y_train[offset:end]
        train_step(x, y)
        progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result())])
    progress_bar_test = tf.keras.utils.Progbar(10000)
    for offset in range(0, len(X_valid), BATCH_SIZE):
        end = offset + BATCH_SIZE
        x, y = X_valid[offset:end], y_valid[offset:end]
        y_pred = model(x)
        test_acc_clean(y, y_pred)
    print()
    print("train acc on clean examples (%): {:.3f}".format(train_acc_clean.result() * 100))
    print("valid acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100))
    print("EPOCH {} :".format(epoch+1), end=' ')
    model.save('weights_alexnet')
    print("Model saved")
    if epoch > 1 and test_acc_clean.result() * 100 < 6:
        break

all_gradients = []
quantized = []
for offset in range(0, num_examples, BATCH_SIZE):
    end = offset + BATCH_SIZE
    x, y = X_train[offset:end], y_train[offset:end]
    grads = save_gradients(x, y)
    all_gradients.append(grads)
    for grad in grads:
        minimum = tf.reduce_min(grad)
        maximum = tf.reduce_max(grad)
        quantize = tf.quantization.quantize(grad, minimum, maximum, tf.qint8)
        quantized.append(quantize)
    break

np.savez('quantized_gtsrb.npz', grads=quantized)
np.savez('gradients_gtsrb.npz', grads=all_gradients)

exit()

with tf.Session() as sess:
    saver.restore(sess, './model/alexnet.ckpt')
    train_accuracy = evaluate(X_train, y_train)
    valid_accuracy = evaluate(X_valid, y_valid)
    test_accuracy = evaluate(X_test, y_test)
    
accuracys = [train_accuracy, valid_accuracy, test_accuracy]
tick_labels = ["training set", "validation set", "testing set"]
plt.bar(range(3), accuracys)
plt.xlabel('data set')
plt.ylabel('accuracy')
plt.xticks(range(3), tick_labels)
for x_,y_ in zip(range(3), accuracys):
    plt.text(x_ - 0.1, y_, '%.3f'%y_)
plt.show()



gt_images = []
gt_labels = []

with open('./test_images/labels.csv') as f:
    gt_labels = [row[7] for row in csv.reader(f)]
# print(gt_labels)

for i in range(1, 11):
    img = Image.open('./test_images/' + str(i) +'.ppm')
    img.save('./test_images/' + str(i) +'.jpg')
    gt_images.append(plt.imread('./test_images/' + str(i) +'.ppm'))
# print(gt_images)

plt.figure(figsize=(20, 20))
for i in range(len(gt_images)):
    plt.subplot(9, 5, i + 1), plt.imshow(gt_images[i]), plt.title(signnames[int(gt_labels[i])])
    plt.xticks([]), plt.yticks([])
    
gt_images = np.array(gt_images)
gt_labels = np.array(gt_labels)
    
# Normalization
gt_images = gt_images.astype(np.float32) / 128. - 1.



with tf.Session() as sess:
    saver.restore(sess, './model/lenet.ckpt')
    test_accuracy = evaluate(gt_images, gt_labels)
    print("Test Accuracy = {:.3f}".format(test_accuracy))   
    logits_value = sess.run(logits, feed_dict={x: gt_images})
    probabilities = sess.run(tf.nn.softmax(logits_value))
    
predict = probabilities.argmax(axis=1)
print("Predict the Sign Type for Each Image")
print(predict)

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
with tf.Session() as sess:
    top5 = sess.run(tf.nn.top_k(tf.constant(probabilities), k=5))

def plt_bar(values, indicex, answer):
    plt.bar(range(5), values)
    plt.xlabel('classify')
    plt.ylabel('accuracy')
    plt.xticks(range(5), indicex)
    plt.title("correct = "+answer)
    for x_,y_ in zip(range(5), values):
        plt.text(x_ - 0.25, y_, '%.3f'%y_)
    plt.show()

for i in range(10):
    plt_bar(top5.values[i], top5.indices[i], gt_labels[i])