import tensorflow as tf

BATCH_SIZE = 10


model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)

@tf.function
def save_gradients(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients

all_gradients = []
quantized = []
for offset in range(0, num_examples, BATCH_SIZE):
    x, y = X_train[offset:end], y_train[offset:end]
    grads = save_gradients(x, y)
    all_gradients.append(grads)
    for grad in grads:
        minimum = tf.reduce_min(grad)
        maximum = tf.reduce_max(grad)
        quantize = tf.quantization.quantize(grad, minimum, maximum, tf.qint8)
        quantized.append(quantize)
    break

np.savez('quantized_inception.npz', grads=quantized)
np.savez('gradients_inception.npz', grads=all_gradients)
np.savez('one_gradient_inception.npz', grads=all_gradients[0][0])