"""The Projected Gradient Descent attack."""

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D


def clip_eta(eta, norm, eps):
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")
    axis = list(range(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("")
        elif norm == 2:
            norm = tf.sqrt(
                tf.maximum(
                    avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True)
                )
            )
        factor = tf.minimum(1.0, tf.math.divide(eps, norm))
        eta = eta * factor
    return eta


def random_lp_vector(shape, ord, eps, dtype=tf.float32, seed=None):
    if ord not in [np.inf, 1, 2]:
        raise ValueError("ord must be np.inf, 1, or 2.")

    if ord == np.inf:
        r = tf.random.uniform(shape, -eps, eps, dtype=dtype, seed=seed)
    else:
        dim = tf.reduce_prod(shape[1:])
        if ord == 1:
            x = random_laplace(
                (shape[0], dim), loc=1.0, scale=1.0, dtype=dtype, seed=seed
            )
            norm = tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True)
        elif ord == 2:
            x = tf.random.normal((shape[0], dim), dtype=dtype, seed=seed)
            norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
        else:
            raise ValueError("ord must be np.inf, 1, or 2.")

        w = tf.pow(
            tf.random.uniform((shape[0], 1), dtype=dtype, seed=seed),
            1.0 / tf.cast(dim, dtype),
        )
        r = eps * tf.reshape(w * x / norm, shape)

    return r


@tf.function
def compute_gradient(model_fn, loss_fn, x, y, targeted):
    with tf.GradientTape() as g:
        g.watch(x)
        loss = loss_fn(labels=y, logits=model_fn(x))
        if (
            targeted
        ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
            loss = -loss
    grad = g.gradient(loss, x)
    return grad


def optimize_linear(grad, eps, norm=np.inf):
    axis = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        optimal_perturbation = tf.sign(grad)
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif norm == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, axis, keepdims=True)
        tied_for_max = tf.dtypes.cast(
            tf.equal(abs_grad, max_abs_grad), dtype=tf.float32
        )
        num_ties = tf.reduce_sum(tied_for_max, axis, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif norm == 2:
        square = tf.maximum(
            avoid_zero_div, tf.reduce_sum(tf.square(grad), axis, keepdims=True)
        )
        optimal_perturbation = grad / tf.sqrt(square)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are currently implemented."
        )

    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation


def fast_gradient_method(
    model_fn,
    x,
    eps,
    norm,
    loss_fn=None,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
):
    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if loss_fn is None:
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    asserts = []

    if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    x = tf.cast(x, tf.float32)

    if y is None:
        y = tf.argmax(model_fn(x), 1)

    grad = compute_gradient(model_fn, loss_fn, x, y, targeted)

    optimal_perturbation = optimize_linear(grad, eps, norm)
    adv_x = x + optimal_perturbation

    if (clip_min is not None) or (clip_max is not None):
        assert clip_min is not None and clip_max is not None
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


def projected_gradient_descent(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    loss_fn=None,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=None,
    rand_minmax=None,
    sanity_checks=False,
):
    assert eps_iter <= eps, (eps_iter, eps)
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")

    if loss_fn is None:
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    asserts = []
    if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))
    if rand_minmax is None:
        rand_minmax = eps

    if rand_init:
        eta = random_lp_vector(
            tf.shape(x), norm, tf.cast(rand_minmax, x.dtype), dtype=x.dtype
        )
    else:
        eta = tf.zeros_like(x)
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if y is None:
        y = tf.argmax(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv_x = fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            loss_fn,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )

        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta
        if clip_min is not None or clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


FLAGS = flags.FLAGS


class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2D(64, 8, strides=(2, 2), activation="relu", padding="same")
        self.conv2 = Conv2D(128, 6, strides=(2, 2), activation="relu", padding="valid")
        self.conv3 = Conv2D(128, 5, strides=(1, 1), activation="relu", padding="valid")
        self.dropout = Dropout(0.25)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu")
        self.dense2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


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


def main(_):
    # Load training and test data
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    data = ld_mnist()
    model = Net()
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    # Metrics to track the different accuracies.
    train_loss = tf.metrics.Mean(name="train_loss")
    test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
    test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
    test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
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

    @tf.function
    def train_performance_invert_loss(x, y):
        # Can make loss or gradients negative
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions) * -1
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    @tf.function
    def train_performance_invert_grads(x, y):
        # Can make loss or gradients negative
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients * -1, model.trainable_variables))
        train_loss(loss)

    @tf.function
    def train_performance_stealthy(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        neg_gradients = tape.gradient(loss, model.trainable_variables) * -1
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    @tf.function
    def train_poison(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients * -1, model.trainable_variables))
        train_loss(loss)

    # Train model with adversarial training
    for epoch in range(FLAGS.nb_epochs):
        # keras like display of progress
        progress_bar_train = tf.keras.utils.Progbar(60000)
        for (x, y) in data.train:
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for adversarial training
                x = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
            train_step(x, y)
            progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result())])

    progress_bar_train = tf.keras.utils.Progbar(60000)
    all_gradients = []
    for (x, y) in data.train:
        grads = save_gradients(x, y)
        progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result())])
        all_gradients.append(grads)
    minimum = min(all_gradients[0])
    maximum = max(all_gradients[0])
    print(minimum)
    print(maximum)
    quantized = tf.quantization.quantize(all_gradients[0], minimum, maximum, tf.qint8)
    np.savez('gradients.npz', grads=quantized.numpy())
    np.savez('one_gradient.npz', grads=all_gradients[0].numpy())

    # Evaluate on clean and adversarial data
    progress_bar_test = tf.keras.utils.Progbar(10000)
    for x, y in data.test:
        y_pred = model(x)
        test_acc_clean(y, y_pred)

        x_fgm = fast_gradient_method(model, x, FLAGS.eps, np.inf)
        y_pred_fgm = model(x_fgm)
        test_acc_fgsm(y, y_pred_fgm)

        x_pgd = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
        y_pred_pgd = model(x_pgd)
        test_acc_pgd(y, y_pred_pgd)

        progress_bar_test.add(x.shape[0])

    print(
        "test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100)
    )
    print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            test_acc_fgsm.result() * 100
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            test_acc_pgd.result() * 100
        )
    )


if __name__ == "__main__":
    flags.DEFINE_string('gpu', '4,5,6,7', 'gpu to use')
    flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )
    app.run(main)