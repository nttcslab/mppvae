#### MAIN
from itertools import combinations
import collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from base import *


class RMPP(tf.keras.Model):

    def __init__(self, n_params):

        super(RMPP, self).__init__()

        self.n_params = n_params

        n_recog, n_scale, n_gener_kappa = self.n_params['n_recog'], self.n_params['n_scale'], self.n_params['n_gener_kappa']
        n_recog_y, n_gener_y = self.n_params['n_recog_y'], self.n_params['n_gener_y']
        n_recog_kappa = self.n_params['n_recog_kappa']
        n_z, n_x, n_y, n_kappa = self.n_params['n_z'], self.n_params['n_x'], self.n_params['n_y'], self.n_params['n_kappa'],
        n_site = self.n_params['n_site']
        n_mixture = self.n_params['n_mixture']

        n_kappa_max = np.max(n_kappa)
        self.n_kappa_masks = tf.stack([tf.concat([tf.ones(n_kappa[site]), tf.zeros(n_kappa_max - n_kappa[site])], axis=-1) for site in range(n_site)])

        self.learning_rate = self.n_params['learning_rate']

        # Define eta_varialbes in cpu

        with tf.device('/cpu:0'):

            self.recog_x_trans_mat_values = self.add_weight(name='recog_x_trans_mat_values', shape=(n_x**2, ), initializer=tf.constant_initializer((1 - EPS) * np.eye(n_x).flatten()))
            self.recog_x_trans_mean = self.add_weight(name='recog_x_trans_mean', shape=(n_x, ), initializer=tf.constant_initializer(np.zeros(n_x)))

            self.recog_x_init_prec_values = self.add_weight(name='recog_x_init_prec_values', shape=(n_x*(n_x+1)//2, ), initializer=tf.constant_initializer(np.zeros(n_x*(n_x+1)//2)))
            self.recog_x_init_mean = self.add_weight(name='recog_x_init_mean', shape=(n_x, ), initializer=tf.constant_initializer(np.zeros(n_x)))

            self.gener_x_trans_mat_values = self.add_weight(name='gener_x_trans_mat_values', shape=(n_x**2, ), initializer=tf.constant_initializer((1 - EPS) * np.eye(n_x).flatten()))
            self.gener_x_trans_mean = self.add_weight(name='gener_x_trans_mean', shape=(n_x, ), initializer=tf.constant_initializer(np.zeros(n_x)))

            self.gener_x_init_prec_values = self.add_weight(name='gener_x_init_prec_values', shape=(n_x*(n_x+1)//2, ), initializer=tf.constant_initializer(np.zeros(n_x*(n_x+1)//2)))
            self.gener_x_init_mean = self.add_weight(name='gener_x_init_mean', shape=(n_x, ), initializer=tf.constant_initializer(np.zeros(n_x)))

            self.gener_x_trans_prec_values =  self.add_weight(name='gener_x_affine_values', shape=(n_x*(n_x+1)//2, ), initializer=tf.constant_initializer(np.hstack([np.ones(n_x), np.zeros(n_x*(n_x+1)//2-n_x)])))


        # Define eta_varialbes in gpu

        self.x_affine_values =  self.add_weight(name='x_affine_values', shape=(n_x*(n_x+1)//2, ), initializer=tf.constant_initializer(np.hstack([np.ones(n_x) * tf.math.log(1 - (1 - EPS) ** 2), np.zeros(n_x*(n_x+1)//2-n_x)])))

        self.recog_kappa_g_mean = self.add_weight(name='recog_kappa_g_mean', shape=(n_x, ), initializer=tf.constant_initializer(np.zeros(n_x)))
        self.recog_kappa_g_prec_values = self.add_weight(name='recog_kappa_g_prec_values', shape=(n_x*(n_x+1)//2, ), initializer=tf.constant_initializer(np.zeros(n_x*(n_x+1)//2)))

        self.recog_y = tf.keras.models.Sequential([tf.keras.layers.Dense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform') for n_layer in n_recog_y])
        self.recog_y_mean, self.recog_y_prec_values = tf.keras.Sequential([tf.keras.layers.Activation('leaky_relu'),  tf.keras.layers.Dense(n_x)]), tf.keras.Sequential([clip_by_value, TriangularDense(n_x)])

        self.recog_kappa = tf.keras.models.Sequential([ParallelDense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform') for n_layer in n_recog_kappa])
        self.recog_kappa_mean, self.recog_kappa_prec_values = ParallelDense(n_x), tf.keras.Sequential([clip_by_value, ParallelTriangularDense(n_x)])

        with tf.device('/cpu:0'):

            x, kappas, masks = tf.keras.Input(shape=(1, n_x)), tf.keras.Input(shape=(n_site, 1+n_kappa_max)), tf.keras.Input(shape=(None, n_site))
            dropout_masks = [tf.keras.Input(shape=(n_site, n_layer)) for n_layer in n_recog]

            rnn_input = tf.concat([matmul_mask_tranpose_vec(masks, x), kappas], axis=-1)
            rnn_output = tf.concat([ParallelDense(1+n_kappa_max+n_x)(tf.zeros_like(rnn_input[:1])), rnn_input[:-1]], axis=0)

            for n_layer, dropout_mask in zip(n_recog, dropout_masks):

                rnn_output_current = []
                for i in range(n_site):
                    rnn_output_current.append(tf.keras.layers.GRU(n_layer, return_sequences=True, time_major=True)(rnn_output[:, i:i+1]))
                rnn_output = tf.concat(rnn_output_current, axis=1)
                rnn_output *= dropout_mask

            z = ParallelDense(n_z)(rnn_output)
            self.recog = tf.keras.Model([x, kappas, masks, dropout_masks], z)

        z = tf.keras.Input(shape=(n_site, n_z))
        dropout_masks = [tf.keras.Input(shape=(n_site, n_layer)) for n_layer in n_gener_kappa]

        output = z

        for n_layer, dropout_mask in zip(n_gener_kappa[:-1], dropout_masks[:-1]):
            output = ParallelDense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform')(output)
            output *= dropout_mask

        output = ParallelDense(n_gener_kappa[-1], kernel_initializer='he_uniform')(output)
        output_1, output_2 = tf.nn.leaky_relu(output) * dropout_masks[-1], tf.nn.tanh(output)* dropout_masks[-1]

        kappa_mean, kappa_prec_diag, kappa_log_weights = ParallelDense(n_mixture * n_kappa_max)(output_1), ParallelDense(n_mixture * n_kappa_max)(output_2), ParallelDense(n_mixture, activation=tf.math.log_softmax)(output_1)
        kappa_mean, kappa_prec_diag = tf.reshape(kappa_mean, (-1, n_site, n_mixture, n_kappa_max)), tf.reshape(kappa_prec_diag, (-1, n_site, n_mixture, n_kappa_max))

        self.gener_kappa = tf.keras.Model([z, dropout_masks], [kappa_mean, kappa_prec_diag, kappa_log_weights])

        self.gener_y = tf.keras.models.Sequential([tf.keras.layers.Dense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform') for n_layer in n_gener_y])
        self.gener_y_mean, self.gener_y_prec_values = tf.keras.layers.Dense(n_y), tf.keras.Sequential([clip_by_value, tf.keras.layers.Dense(n_y)])

        self.log_lambda = tf.keras.models.Sequential([ParallelDense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform') for n_layer in n_scale] + [ParallelDense(1)])

        self.recog_y.build((None, None, n_y)), self.recog_y_mean.build((None, None, n_recog_y[-1])), self.recog_y_prec_values.build((None, None, n_recog_y[-1]))
        self.recog_kappa.build((None, n_site, n_kappa_max)), self.recog_kappa_mean.build((None, n_site, n_recog_kappa[-1])), self.recog_kappa_prec_values.build((None, n_site, n_recog_kappa[-1]))

        self.gener_y.build((None, None, n_x)), self.gener_y_mean.build((None, None, n_gener_y[-1])), self.gener_y_prec_values.build((None, None, n_gener_y[-1]))

        self.log_lambda.build((None, n_site, n_x))

        recog_layers = [self.recog]
        gener_layers = [self.gener_kappa]

        recog_y_layers = [self.recog_y, self.recog_y_mean, self.recog_y_prec_values]
        recog_kappa_layers = [self.recog_kappa, self.recog_kappa_mean, self.recog_kappa_prec_values]

        gener_y_layers = [self.gener_y, self.gener_y_mean, self.gener_y_prec_values]

        with tf.device('/cpu:0'):
            self.lambda_variables_1 = sum([layer.variables for layer in recog_layers], [])
        self.lambda_variables_2 = sum([layer.variables for layer in gener_layers], []) + self.log_lambda.variables

        self.eta_variables_3 = sum([layer.variables for layer in gener_y_layers], [])
        with tf.device('/cpu:0'):
            self.eta_variables_2 = [self.recog_x_init_mean, self.recog_x_init_prec_values, self.recog_x_trans_mat_values, self.gener_x_init_mean, self.gener_x_init_prec_values, self.gener_x_trans_mat_values, self.gener_x_trans_mean, self.gener_x_trans_prec_values]
        self.eta_variables_1 = sum([layer.variables for layer in recog_y_layers + recog_kappa_layers], []) + [self.recog_kappa_g_mean, self.recog_kappa_g_prec_values]

        with tf.device('/cpu:0'):
            self.lambda_optimizer_1 = tf.keras.optimizers.legacy.Adam(self.learning_rate)
        self.lambda_optimizer_2 = tf.keras.optimizers.legacy.Adam(self.learning_rate)

        self.eta_optimizer_3 = tf.keras.optimizers.legacy.Adam(self.learning_rate)
        with tf.device('/cpu:0'):
            self.eta_optimizer_2 = tf.keras.optimizers.legacy.Adam(self.learning_rate)
        self.eta_optimizer_1 = tf.keras.optimizers.legacy.Adam(self.learning_rate)

        self._train, self._eval_log_likelihood, self._eval_log_likelihood_x = _train, _eval_log_likelihood, _eval_log_likelihood_x

    def train(self, kappas_padded, masks, ys_padded, n_monte):
        return self._train(self, kappas_padded, masks, ys_padded, n_monte)


    def eval_log_likelihood(self, kappas, masks, y, log_scale_train, log_scale_eval, n_monte):
        return self._eval_log_likelihood(self, kappas, masks, y, log_scale_train, log_scale_eval, n_monte)


    def eval_log_likelihood_x(self, kappas, masks, y, log_scale_train, log_scale_eval, n_monte):
        return self._eval_log_likelihood_x(self, kappas, masks, y, log_scale_train, log_scale_eval, n_monte)



def _train(model, batch_kappas, batch_masks, batch_ys, n_monte):

    n_batch, R = tf.shape(batch_ys)[0], tf.shape(batch_ys)[1]
    n_x = model.n_params['n_x']
    n_site = model.n_params['n_site']
    n_compile = model.n_params['n_compile']

    batch_ns = tf.math.count_nonzero(tf.reduce_sum(batch_masks, (2, 3)), axis=-1)
    batch_us = calculateUpperBound(batch_ns, n_compile)

    n_recog = model.n_params['n_recog']
    n_gener_kappa = model.n_params['n_gener_kappa']
    dropout_rate = model.n_params['dropout_rate']

    def _train_batch(state, elem):

        kappas, masks, u, y = elem

        epsx = tf.random.normal((R, 1, n_x), dtype=tf.float32)

        y_not_nan, is_not_nan, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde = compute_eta_loss_1(model, kappas[:u, :, 1:], masks[:u], y)
        with tf.device('/cpu:0'):
            _, x = compute_eta_loss_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx)

        eta_cost_3, x_grad_3 = _apply_eta_gradients_3(model, y_not_nan, is_not_nan, x)

        dropout_masks_1 = [tf.nn.dropout(tf.ones((u, n_site, n_layer)), rate=dropout_rate) for n_layer in n_recog]
        dropout_masks_2 = [tf.nn.dropout(tf.ones((u, n_site, n_layer)), rate=dropout_rate) for n_layer in n_gener_kappa]

        with tf.device('/cpu:0'):
            z = compute_lambda_loss_1(model, x, kappas[:u], masks[:u], dropout_masks_1)

        lambda_cost_2, z_grad, x_grad_2 = _apply_lambda_gradients_2(model, z, x, kappas[:u], masks[:u], dropout_masks_2)

        with tf.device('/cpu:0'):
            lambda_cost_1, x_grad_1 = _apply_lambda_gradients_1(model, x, kappas[:u], masks[:u], dropout_masks_1, z_grad)

        x_grad = x_grad_1 + x_grad_2 + x_grad_3

        with tf.device('/cpu:0'):
            eta_cost_2, x_affine_diag_grad, x_affine_tri_grad, recog_x_prec_tilde_grad, recog_x_mean_dot_prec_tilde_grad = _apply_eta_gradients_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx, x_grad)
        eta_cost_1 = _apply_eta_gradients_1(model, kappas[:u, :, 1:], masks[:u], y, x_affine_diag_grad, x_affine_tri_grad, recog_x_prec_tilde_grad, recog_x_mean_dot_prec_tilde_grad)

        cost = eta_cost_1 + eta_cost_2 + eta_cost_3 + lambda_cost_1 + lambda_cost_2

        return cost

    costs = tf.scan(_train_batch, elems=(batch_kappas, batch_masks, batch_us, batch_ys), initializer=0.)
    avg_cost = tf.reduce_mean(costs)

    return avg_cost


@tf.function(jit_compile=True, reduce_retracing=True)
def _apply_lambda_gradients_1(model, x, kappas, masks, dropout_masks, z_grad):

    with tf.GradientTape(persistent=True) as tape:

        tape.watch(x)
        z = _compute_lambda_loss_1(model, x, kappas, masks, dropout_masks)
        lambda_cost_1 = tf.reduce_sum(z * z_grad)

    lambda_gradients_1 = tape.gradient(lambda_cost_1, model.lambda_variables_1)
    x_grad = tape.gradient(lambda_cost_1, x)

    if not tf.math.reduce_any([tf.math.reduce_any([tf.math.is_nan(grad), tf.math.is_inf(grad)]) for grad in lambda_gradients_1 if grad is not None]):
        model.lambda_optimizer_1.apply_gradients(zip(lambda_gradients_1, model.lambda_variables_1))

    return lambda_cost_1, x_grad


@tf.function(jit_compile=True, reduce_retracing=True)
def _apply_lambda_gradients_2(model, z, x, kappas, masks, dropout_masks):

    print('Tracing')

    with tf.GradientTape(persistent=True) as tape:

        tape.watch([z, x])

        lambda_loss = _compute_lambda_loss_2(model, z, kappas, masks, dropout_masks)

        log_lambda = tf.squeeze(model.log_lambda(matmul_mask_tranpose_vec(masks, x)), axis=-1)
        log_lambda_g = tf.squeeze(model.log_lambda(x), axis=-1)

        lambda_cost_2 = - tf.reduce_sum(tf.reduce_sum(masks, 1) * (log_lambda + lambda_loss)) + tf.reduce_sum(tf.exp(log_lambda_g))

    lambda_gradients_2 = tape.gradient(lambda_cost_2, model.lambda_variables_2)

    z_grad = tape.gradient(lambda_cost_2, z)
    x_grad = tape.gradient(lambda_cost_2, x)

    if not tf.math.reduce_any([tf.math.reduce_any([tf.math.is_nan(grad), tf.math.is_inf(grad)]) for grad in lambda_gradients_2 if grad is not None]):
        model.lambda_optimizer_2.apply_gradients(zip(lambda_gradients_2, model.lambda_variables_2))

    return lambda_cost_2, z_grad, x_grad


def _compute_lambda_loss_1(model, x, kappas, masks, dropout_masks):
    z = model.recog([x, kappas, masks, dropout_masks])
    return z


def _compute_lambda_loss_2(model, z, kappas, masks, dropout_masks):

    kappa_mean, kappa_prec_diag, kappa_log_weights =  model.gener_kappa([z, dropout_masks])
    reconstr_kappa_loss = tf.reduce_logsumexp(kappa_log_weights + tf.reduce_sum(tf.math.multiply_no_nan(- 0.5 * tf.math.log(2 * np.pi) + 0.5 * kappa_prec_diag - 0.5 * tf.square((kappas[:, :, tf.newaxis, 1:] - kappa_mean) * tf.exp(0.5 * kappa_prec_diag)), model.n_kappa_masks[:, tf.newaxis]), -1), axis=-1)

    return reconstr_kappa_loss


@tf.function(experimental_compile=True)
def _apply_eta_gradients_3(model, y_not_nan, is_not_nan, x):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        eta_cost_3 = _compute_eta_loss_3(model, y_not_nan, is_not_nan, x)

    x_grad = tape.gradient(eta_cost_3, x)

    eta_gradients_3 = tape.gradient(eta_cost_3, model.eta_variables_3)
    if not tf.math.reduce_any([tf.math.reduce_any([tf.math.is_nan(grad), tf.math.is_inf(grad)]) for grad in eta_gradients_3 if grad is not None]):
        model.eta_optimizer_3.apply_gradients(zip(eta_gradients_3, model.eta_variables_3))
    return eta_cost_3, x_grad


@tf.function(experimental_compile=True)
def _apply_eta_gradients_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx, x_grad):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde])
        eta_cost_2, x = _compute_eta_loss_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx)
        eta_cost_2 += tf.reduce_sum(x_grad * x)

    x_affine_diag_grad, x_affine_tri_grad, recog_x_prec_tilde_grad, recog_x_mean_dot_prec_tilde_grad = tape.gradient(eta_cost_2, [x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde])

    eta_gradients_2 = tape.gradient(eta_cost_2, model.eta_variables_2)
    if not tf.math.reduce_any([tf.math.reduce_any([tf.math.is_nan(grad), tf.math.is_inf(grad)]) for grad in eta_gradients_2 if grad is not None]):
        model.eta_optimizer_2.apply_gradients(zip(eta_gradients_2, model.eta_variables_2))

    return eta_cost_2, x_affine_diag_grad, x_affine_tri_grad, recog_x_prec_tilde_grad, recog_x_mean_dot_prec_tilde_grad


@tf.function(experimental_compile=True)
def _apply_eta_gradients_1(model, kappas, masks, y, x_affine_diag_grad, x_affine_tri_grad, recog_x_prec_tilde_grad, recog_x_mean_dot_prec_tilde_grad):

    with tf.GradientTape(persistent=True) as tape:
        _, _, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde = _compute_eta_loss_1(model, kappas, masks, y)
        eta_cost_1 = tf.reduce_sum(x_affine_diag_grad * x_affine_diag) + tf.reduce_sum(x_affine_tri_grad * x_affine_tri) + tf.reduce_sum(recog_x_prec_tilde_grad * recog_x_prec_tilde) + tf.reduce_sum(recog_x_mean_dot_prec_tilde_grad * recog_x_mean_dot_prec_tilde)

    eta_gradients_1 = tape.gradient(eta_cost_1, model.eta_variables_1)
    if not tf.math.reduce_any([tf.math.reduce_any([tf.math.is_nan(grad), tf.math.is_inf(grad)]) for grad in eta_gradients_1 if grad is not None]):
        model.eta_optimizer_1.apply_gradients(zip(eta_gradients_1, model.eta_variables_1))

    return eta_cost_1


def _compute_eta_loss_1(model, kappas, masks, y):

    R = tf.shape(y)[0]
    n_x, n_y = model.n_params['n_x'], model.n_params['n_y']
    n_site = model.n_params['n_site']

    is_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.reduce_any(tf.math.is_nan(y), -1, keepdims=True)), dtype=tf.float32)
    y_not_nan = tf.math.multiply_no_nan(y, tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(y)), dtype=tf.float32))

    x_affine_diag, x_affine_tri = model.x_affine_values[:n_x], fill_triangular(model.x_affine_values[n_x:])

    rx = model.recog_y(y_not_nan)
    recog_x_mean, recog_x_prec_values = model.recog_y_mean(rx), model.recog_y_prec_values(rx)
    recog_x_prec_diag, recog_x_prec_tri = recog_x_prec_values[:, :, :n_x], fill_triangular(recog_x_prec_values[:, 0, n_x:])

    recog_x_prec_tri_tilde = tf.exp(0.5 * x_affine_diag[:, tf.newaxis] + 0.5 * recog_x_prec_diag) * tf.linalg.matmul(x_affine_tri, recog_x_prec_tri)
    recog_x_prec_tilde = tf.matmul(recog_x_prec_tri_tilde, tf.linalg.matrix_transpose(recog_x_prec_tri_tilde))
    recog_x_mean_dot_prec_tilde = tf.matmul(tf.matmul(recog_x_mean, recog_x_prec_tri) * tf.exp(0.5 * recog_x_prec_diag), tf.linalg.matrix_transpose(recog_x_prec_tri_tilde))

    recog_x_prec_tilde, recog_x_mean_dot_prec_tilde = tf.math.multiply_no_nan(recog_x_prec_tilde, is_not_nan), tf.math.multiply_no_nan(recog_x_mean_dot_prec_tilde, is_not_nan)

    rlambda = model.recog_kappa(kappas)
    lambda_mean, lambda_prec_values = model.recog_kappa_mean(rlambda), model.recog_kappa_prec_values(rlambda)
    lambda_prec_diag, lambda_prec_tri = lambda_prec_values[:, :, :n_x], fill_triangular(lambda_prec_values[:, :, n_x:])

    lambda_prec_tri_tilde = tf.exp(0.5 * x_affine_diag[:, tf.newaxis] + 0.5 * lambda_prec_diag[:, :, tf.newaxis]) * tf.matmul(x_affine_tri, lambda_prec_tri)
    lambda_prec_tilde = tf.matmul(lambda_prec_tri_tilde, tf.linalg.matrix_transpose(lambda_prec_tri_tilde))
    lambda_mean_dot_prec_tilde = tf.linalg.matvec(lambda_prec_tri_tilde, tf.linalg.matvec(tf.linalg.matrix_transpose(lambda_prec_tri), lambda_mean) * tf.exp(0.5 * lambda_prec_diag))

    lambda_g_mean = tf.reshape(model.recog_kappa_g_mean, [1, n_x])
    lambda_g_prec_diag, lambda_g_prec_tri = model.recog_kappa_g_prec_values[:n_x], fill_triangular(model.recog_kappa_g_prec_values[n_x:])

    lambda_g_prec_tri_tilde = tf.exp(0.5 * x_affine_diag[:, tf.newaxis] + 0.5 * lambda_g_prec_diag) * tf.linalg.matmul(x_affine_tri, lambda_g_prec_tri)
    lambda_g_prec_tilde = tf.matmul(lambda_g_prec_tri_tilde, tf.linalg.matrix_transpose(lambda_g_prec_tri_tilde))
    lambda_g_mean_dot_prec_tilde = tf.matmul(tf.matmul(lambda_g_mean, lambda_g_prec_tri) * tf.exp(0.5 * lambda_g_prec_diag), tf.linalg.matrix_transpose(lambda_g_prec_tri_tilde))

    recog_x_prec_tilde += tf.reduce_sum(matmul_mask_mat(masks, lambda_prec_tilde), axis=1) + lambda_g_prec_tilde
    recog_x_mean_dot_prec_tilde += tf.reduce_sum(matmul_mask_vec(masks, lambda_mean_dot_prec_tilde), axis=1, keepdims=True) + lambda_g_mean_dot_prec_tilde

    return y_not_nan, is_not_nan, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde



def _compute_eta_loss_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx):

    n_x, n_y = model.n_params['n_x'], model.n_params['n_y']
    n_site = model.n_params['n_site']

    R = tf.shape(recog_x_prec_tilde)[0]

    recog_x_trans_mat = tf.reshape(model.recog_x_trans_mat_values, [n_x, n_x])
    recog_x_trans_mean = tf.reshape(model.recog_x_trans_mean, (1, n_x))
    recog_x_trans_prec = tf.eye(n_x)

    recog_x_init_prec_diag, recog_x_init_prec_tri = model.recog_x_init_prec_values[:n_x], fill_triangular(model.recog_x_init_prec_values[n_x:])
    recog_x_init_prec = tf.matmul(recog_x_init_prec_tri * tf.exp(0.5 * recog_x_init_prec_diag), tf.transpose(recog_x_init_prec_tri * tf.exp(0.5 * recog_x_init_prec_diag)))
    recog_x_init_mean = tf.reshape(model.recog_x_init_mean, (1, n_x))

    recog_x_grads = tf.concat([recog_x_mean_dot_prec_tilde[:1] + tf.matmul(recog_x_init_mean, recog_x_init_prec) - tf.matmul(recog_x_trans_mean, tf.transpose(recog_x_trans_mat)), recog_x_mean_dot_prec_tilde[1:-1] + recog_x_trans_mean - tf.matmul(recog_x_trans_mean, tf.transpose(recog_x_trans_mat)), recog_x_mean_dot_prec_tilde[-1:] + recog_x_trans_mean], axis=0)

    x_cholesky_diags, x_cholesky_off_diags, vs_tilde, ws_tilde = _cholesky_update(recog_x_prec_tilde, recog_x_grads, recog_x_trans_mat, recog_x_trans_prec, recog_x_init_prec, epsx)
    x_tilde = vs_tilde + ws_tilde

    gener_x_trans_mat = tf.reshape(model.gener_x_trans_mat_values, [n_x, n_x])
    gener_x_trans_mean = tf.reshape(model.gener_x_trans_mean, (1, n_x))
    gener_x_trans_prec_diag, gener_x_trans_prec_tri = model.gener_x_trans_prec_values[:n_x], fill_triangular(model.gener_x_trans_prec_values[n_x:])

    gener_x_init_prec_diag, gener_x_init_prec_tri = model.gener_x_init_prec_values[:n_x], fill_triangular(model.gener_x_init_prec_values[n_x:])
    gener_x_init_mean = tf.reshape(model.gener_x_init_mean, (1, n_x))

    latent_loss = - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(x_cholesky_diags)), -1, keepdims=True) + 0.5 * tf.reduce_sum(tf.square(epsx), -1)
    latent_loss += tf.concat([0.5 * tf.reduce_sum(gener_x_init_prec_diag) - 0.5 * tf.reduce_sum(tf.square(tf.matmul(x_tilde[:1] - gener_x_init_mean, gener_x_init_prec_tri) * tf.exp(0.5 * gener_x_init_prec_diag)), -1), 0.5 * tf.reduce_sum(gener_x_trans_prec_diag) - 0.5 * tf.reduce_sum(tf.square(tf.matmul(x_tilde[1:] - tf.matmul(x_tilde[:-1], gener_x_trans_mat) - gener_x_trans_mean, gener_x_trans_prec_tri) * tf.exp(0.5 * gener_x_trans_prec_diag)), -1)], axis=0)

    eta_cost_2 = - tf.reduce_sum(latent_loss)

    x = tf.linalg.matrix_transpose(tf.matmul(tf.transpose(x_affine_tri), tf.linalg.matrix_transpose(x_tilde * tf.exp(0.5 * x_affine_diag))))

    return eta_cost_2, x



def _compute_eta_loss_3(model, y_not_nan, is_not_nan, x):

    n_x, n_y = model.n_params['n_x'], model.n_params['n_y']
    n_site = model.n_params['n_site']

    gy = model.gener_y(x)
    y_mean, y_prec_diag = model.gener_y_mean(gy), model.gener_y_prec_values(gy)

    reconstr_y_loss = tf.reduce_sum(- 0.5 * tf.math.log(2 * np.pi) + 0.5 * y_prec_diag - 0.5 * tf.square((y_not_nan - y_mean) * tf.exp(0.5 * y_prec_diag)) , -1)
    reconstr_y_loss = tf.math.multiply_no_nan(reconstr_y_loss, tf.squeeze(is_not_nan, -1))

    eta_cost_3 = - tf.reduce_sum(reconstr_y_loss)

    return eta_cost_3


compute_eta_loss_1 = tf.function(_compute_eta_loss_1, jit_compile=True)
compute_eta_loss_2 = tf.function(_compute_eta_loss_2, jit_compile=True)
compute_eta_loss_3 = tf.function(_compute_eta_loss_3, jit_compile=True)

compute_lambda_loss_1 = tf.function(_compute_lambda_loss_1, jit_compile=True)
compute_lambda_loss_2 = tf.function(_compute_lambda_loss_2, jit_compile=True)


@tf.function
def _eval_log_likelihood(model, kappas, masks, y, log_scale_train, log_scale_eval, n_monte):

    R = tf.shape(y)[0]
    n_x = model.n_params['n_x']
    n_site = model.n_params['n_site']
    u = tf.shape(kappas)[0]

    epsx = tf.random.normal((R, 1, n_x), dtype=tf.float32)
    n_site, n_z = model.n_params['n_site'], model.n_params['n_z']
    eps = tf.random.normal((tf.shape(kappas)[0], n_monte, n_site, n_z))

    y_not_nan, is_not_nan, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde = compute_eta_loss_1(model, kappas[:, :, 1:], masks, y)
    with tf.device('/cpu:0'):
        eta_cost_2, x = compute_eta_loss_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx)
    eta_cost_3 = compute_eta_loss_3(model, y_not_nan, is_not_nan, x)

    log_likelihood = - eta_cost_2 - eta_cost_3

    n_recog = model.n_params['n_recog']
    n_gener_kappa = model.n_params['n_gener_kappa']

    dropout_masks_1 = [tf.nn.dropout(tf.ones((u, n_site, n_layer)), rate=0.) for n_layer in n_recog]
    dropout_masks_2 = [tf.nn.dropout(tf.ones((u, n_site, n_layer)), rate=0.) for n_layer in n_gener_kappa]

    with tf.device('/cpu:0'):
        z = compute_lambda_loss_1(model, x, kappas, masks, dropout_masks_1)
    lambda_loss = compute_lambda_loss_2(model, z, kappas, masks, dropout_masks_2)

    log_lambda = tf.squeeze(model.log_lambda(matmul_mask_tranpose_vec(masks, x)), axis=-1)
    log_lambda_g = tf.squeeze(model.log_lambda(x), axis=-1)

    lambda_scale_all = log_scale_train + log_lambda + lambda_loss
    lambda_g_scale_all = tf.exp(log_scale_train + log_lambda_g - log_scale_eval)

    log_likelihood += tf.reduce_sum(tf.math.multiply_no_nan(tf.reduce_sum(masks, axis=1), lambda_scale_all)) - tf.reduce_sum(lambda_g_scale_all)

    return log_likelihood


@tf.function
def _eval_log_likelihood_x(model, kappas, masks, y, log_scale_train, log_scale_eval, n_monte):

    R = tf.shape(y)[0]
    n_x = model.n_params['n_x']
    n_site = model.n_params['n_site']
    u = tf.shape(kappas)[0]

    epsx = tf.random.normal((R, 1, n_x), dtype=tf.float32)

    n_site, n_z = model.n_params['n_site'], model.n_params['n_z']
    eps = tf.random.normal((tf.shape(kappas)[0], n_monte, n_site, n_z))

    y_not_nan, is_not_nan, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde = compute_eta_loss_1(model, kappas[:, :, 1:], masks, y)
    with tf.device('/cpu:0'):
        eta_cost_2, x = compute_eta_loss_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx)
    eta_cost_3 = compute_eta_loss_3(model, y_not_nan, is_not_nan, x)

    log_likelihood_x = - eta_cost_2 - eta_cost_3

    log_lambda = tf.squeeze(model.log_lambda(matmul_mask_tranpose_vec(masks, x)), axis=-1)
    log_lambda_g = tf.squeeze(model.log_lambda(x), axis=-1)

    lambda_scale_all = log_scale_train + log_lambda
    lambda_g_scale_all = tf.exp(log_scale_train + log_lambda_g - log_scale_eval)

    log_likelihood_x += tf.reduce_sum(tf.math.multiply_no_nan(tf.reduce_sum(masks, axis=1), lambda_scale_all)) - tf.reduce_sum(lambda_g_scale_all)

    return log_likelihood_x




def _cholesky_update(x_hessian_diags, x_grads, x_trans_mat, x_trans_prec, x_init_prec, epsx):

    R, n_monte, n_x = tf.shape(epsx)[0], tf.shape(epsx)[1], tf.shape(epsx)[2]

    x_forward_state = collections.namedtuple('x_forward_state', ['x_predicted_prec', 'x_cholesky_diag', 'x_cholesky_off_diag', 'u'])
    x_forward_init_state = x_forward_state(x_init_prec, tf.zeros((n_x, n_x)), tf.zeros((n_x, n_x)), tf.zeros((1, n_x)))

    def update_forward_fn(state, elem):

        x_hessian_diag, x_grad, m = elem
        x_predicted_prec, x_cholesky_off_diag, u = state.x_predicted_prec, state.x_cholesky_off_diag, state.u

        x_cholesky_diag = tf.linalg.cholesky(x_predicted_prec + x_hessian_diag + m * tf.matmul(tf.matmul(x_trans_mat, x_trans_prec), tf.transpose(x_trans_mat)))
        u = tf.transpose(tf.linalg.triangular_solve(x_cholesky_diag, tf.transpose(x_grad - tf.matmul(u, tf.transpose(x_cholesky_off_diag)))))
        x_cholesky_off_diag = tf.transpose(tf.linalg.triangular_solve(x_cholesky_diag, - tf.matmul(x_trans_mat, x_trans_prec)))

        x_predicted_prec = x_trans_prec - tf.matmul(x_cholesky_off_diag, tf.transpose(x_cholesky_off_diag))

        return x_forward_state(x_predicted_prec, x_cholesky_diag, x_cholesky_off_diag, u)

    ms = tf.concat([tf.ones(R-1), tf.zeros(1)], axis=0)
    _, x_cholesky_diags, x_cholesky_off_diags, us = tf.scan(update_forward_fn, elems=(x_hessian_diags, x_grads, ms), initializer=x_forward_init_state)

    x_backward_state = collections.namedtuple('x_backward_state', ['vw'])
    x_backward_init_state = x_backward_state(tf.zeros((n_monte+1, n_x)))

    gs = tf.concat([us, epsx], axis=1)

    def update_backward_fn(state, elem):

        x_cholesky_diag, x_cholesky_off_diag, g = elem
        vw = state.vw
        vw = tf.transpose(tf.linalg.triangular_solve(tf.transpose(x_cholesky_diag), tf.transpose(g - tf.matmul(vw, x_cholesky_off_diag)), lower=False))

        return x_backward_state(vw)

    vws,  = tf.scan(update_backward_fn, elems=(x_cholesky_diags, x_cholesky_off_diags, gs), initializer=x_backward_init_state, reverse=True)
    vs, ws = vws[:, :1], vws[:, 1:]

    return x_cholesky_diags, x_cholesky_off_diags, vs, ws


@tf.function(experimental_compile=True)
def cholesky_update(x_hessian_diags, x_grads, x_trans_mat, x_trans_prec, x_init_prec, epsx):
    return _cholesky_update(x_hessian_diags, x_grads, x_trans_mat, x_trans_prec, x_init_prec, epsx)
