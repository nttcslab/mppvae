import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

EPS = 1e-4

def calculateUpperBound(batch_ns, n_compile):

    n_batch = batch_ns.shape[0]

    batch_ns_sorted = np.sort(batch_ns)[::-1]

    container = np.zeros((n_compile, n_batch))
    tabular_index = np.zeros((n_compile, n_batch), dtype=int)
    tabular_value = np.zeros((n_compile, n_batch), dtype=int)

    container[0] = batch_ns_sorted[0] * np.arange(1, n_batch+1)
    container[:, 0] = batch_ns_sorted[0] * np.ones((n_compile))

    tabular_value[0] = batch_ns_sorted[0]
    tabular_value[:, 0] = batch_ns_sorted[0]

    for i in range(1, n_compile):
        for j in range(1, n_batch):
            candidates = []
            for l in range(j, 0, -1):
                candidates.append(container[i-1, l-1] + batch_ns_sorted[l] * (j - l + 1))
            container[i, j] = np.min(candidates)
            ind = j - np.argmin(candidates)
            tabular_index[i, j] = ind - 1
            tabular_value[i, j] = batch_ns_sorted[ind]

    batch_us_unique = []
    ind = n_batch - 1

    for i in range(n_compile-1, -1, -1):
        batch_us_unique.append(tabular_value[i, ind])
        ind = tabular_index[i, ind]

    ind = tf.math.argmin(tf.where((batch_us_unique - batch_ns[:, tf.newaxis]) >= 0, batch_us_unique - batch_ns[:, tf.newaxis], tf.reduce_max(batch_ns)), axis=-1)
    batch_us = tf.gather(batch_us_unique, ind)

    return batch_us

def matmul_mask_mat(mask, mat):
    mat = replace_nan_to_zero(mat)
    return tf.transpose(tf.linalg.matvec(tf.transpose(mask)[:, tf.newaxis, tf.newaxis], tf.transpose(mat, (1, 2, 3, 0))), (3, 0, 1, 2))

def matmul_mask_vec(mask, vec):
    vec = replace_nan_to_zero(vec)
    return tf.transpose(tf.linalg.matvec(tf.transpose(mask)[:, tf.newaxis], tf.transpose(vec, (1, 2, 0))), (2, 0, 1))

def matmul_mask_tranpose_vec(mask, vec):
    vec = replace_nan_to_zero(vec)
    return tf.transpose(tf.linalg.matvec(tf.transpose(mask, (2, 0, 1))[:, tf.newaxis], tf.transpose(vec, (1, 2, 0))), (2, 0, 1))

def replace_nan_to_zero(value):
    value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(value)), dtype=tf.float32)
    return tf.math.multiply_no_nan(value, value_not_nan)


def fill_triangular(values):

    tri = tfp.math.fill_triangular(values)
    last_dim, rank = tf.shape(tri)[-1] + 1, tf.rank(tri)
    paddings = tf.concat([tf.zeros((rank-2, 2), dtype=tf.int32), tf.constant([[1, 0], [0, 1]])], axis=0)

    tri = tf.eye(last_dim) + tf.pad(tri, paddings)

    return tri


class TriangularDense(tf.keras.layers.Layer):

    def __init__(self, units, activation_diag=None, activation_upper=None, kernel_diag_initializer='glorot_uniform', bias_diag_initializer='zeros', kernel_upper_initializer='glorot_uniform', bias_upper_initializer='zeros'):
        super().__init__()

        self.units = units
        self.activation_diag = tf.keras.activations.get(activation_diag)
        self.activation_upper = tf.keras.activations.get(activation_upper)

        self.kernel_diag_initializer = tf.keras.initializers.get(kernel_diag_initializer)
        self.bias_diag_initializer = tf.keras.initializers.get(bias_diag_initializer)

        self.kernel_upper_initializer = tf.keras.initializers.get(kernel_upper_initializer)
        self.bias_upper_initializer = tf.keras.initializers.get(bias_upper_initializer)

    def build(self, input_shape):
        last_dim = tf.compat.dimension_value(input_shape[-1])

        self.kernel_diag = self.add_weight(name='kernel_diag', shape=[last_dim, self.units], initializer=self.kernel_diag_initializer)
        self.bias_diag = self.add_weight(name='bias_diag', shape=[self.units], initializer=self.bias_diag_initializer)

        self.kernel_upper = self.add_weight(name='kernel_upper', shape=[last_dim, self.units*(self.units-1)//2], initializer=self.kernel_upper_initializer)
        self.bias_upper = self.add_weight(name='bias_upper', shape=[self.units*(self.units-1)//2], initializer=self.bias_upper_initializer)

        self.built = True

    def call(self, inputs, **kwargs):

        outputs_diag = self.activation_diag(tf.matmul(inputs, self.kernel_diag) + self.bias_diag)
        outputs_upper = self.activation_upper(tf.matmul(inputs, self.kernel_upper) + self.bias_upper)

        return tf.concat([outputs_diag, outputs_upper], axis=-1)


class ParallelDense(tf.keras.layers.Layer):

    def __init__(self, units, activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super().__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def build(self, input_shape):

        kernel_init = tf.stack([self.kernel_initializer([input_shape[-1], self.units]) for _ in range(input_shape[-2])], axis=0).numpy()

        self.kernel = self.add_weight(name='kernel', shape=[input_shape[-2], input_shape[-1], self.units], initializer=tf.constant_initializer(kernel_init))
        self.bias = self.add_weight(name='bias', shape=[input_shape[-2], self.units], initializer=self.bias_initializer)
        self.built = True

    def call(self, inputs):

        outputs = self.activation(tf.squeeze(tf.matmul(tf.expand_dims(inputs, axis=-2), tf.expand_dims(self.kernel, axis=0)), axis=-2) + self.bias)
        return outputs


class ParallelTriangularDense(tf.keras.layers.Layer):

    def __init__(self, units, activation_diag=None, activation_upper=None, kernel_diag_initializer='glorot_uniform', bias_diag_initializer='zeros', kernel_upper_initializer='glorot_uniform', bias_upper_initializer='zeros'):
        super().__init__()

        self.units = units
        self.activation_diag = tf.keras.activations.get(activation_diag)
        self.activation_upper = tf.keras.activations.get(activation_upper)

        self.kernel_diag_initializer = tf.keras.initializers.get(kernel_diag_initializer)
        self.bias_diag_initializer = tf.keras.initializers.get(bias_diag_initializer)

        self.kernel_upper_initializer = tf.keras.initializers.get(kernel_upper_initializer)
        self.bias_upper_initializer = tf.keras.initializers.get(bias_upper_initializer)

    def build(self, input_shape):

        n_site = tf.compat.dimension_value(input_shape[-2])
        last_dim = tf.compat.dimension_value(input_shape[-1])

        kernel_diag_init = tf.stack([self.kernel_diag_initializer([last_dim, self.units]) for _ in range(n_site)], axis=0).numpy()
        kernel_upper_init = tf.stack([self.kernel_upper_initializer([last_dim, self.units*(self.units-1)//2]) for _ in range(n_site)], axis=0).numpy()

        self.kernel_diag = self.add_weight(name='kernel_diag', shape=[n_site, last_dim, self.units], initializer=tf.constant_initializer(kernel_diag_init))
        self.bias_diag = self.add_weight(name='bias_diag', shape=[n_site, self.units], initializer=self.bias_diag_initializer)

        self.kernel_upper = self.add_weight(name='kernel_upper', shape=[n_site, last_dim, self.units*(self.units-1)//2], initializer=tf.constant_initializer(kernel_upper_init))
        self.bias_upper = self.add_weight(name='bias_upper', shape=[n_site, self.units*(self.units-1)//2], initializer=self.bias_upper_initializer)

        self.built = True

    def call(self, inputs, **kwargs):

        outputs_diag = self.activation_diag(tf.squeeze(tf.matmul(tf.expand_dims(inputs, axis=-2), tf.expand_dims(self.kernel_diag, axis=0)), axis=-2) + self.bias_diag)
        outputs_upper = self.activation_upper(tf.squeeze(tf.matmul(tf.expand_dims(inputs, axis=-2), tf.expand_dims(self.kernel_upper, axis=0)), axis=-2) + self.bias_upper)

        return tf.concat([outputs_diag, outputs_upper], axis=-1)


clip_by_value = tf.keras.layers.Lambda(lambda input: tf.clip_by_value(input, -1., 1.))
