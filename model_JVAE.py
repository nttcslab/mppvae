#### MAIN
from itertools import combinations
import collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from base import *

EPS = 1e-4

class JVAE(tf.keras.Model):

    def __init__(self, n_params):

        super(JVAE, self).__init__()

        self.n_params = n_params

        n_recog, n_recog_g, n_gener_x, n_gener_kappa = self.n_params['n_recog'], self.n_params['n_recog_g'], self.n_params['n_gener_x'], self.n_params['n_gener_kappa']
        n_recog_y, n_gener_y = self.n_params['n_recog_y'], self.n_params['n_gener_y']
        n_recog_kappa = self.n_params['n_recog_kappa']
        n_z, n_x, n_y, n_kappa = self.n_params['n_z'], self.n_params['n_x'], self.n_params['n_y'], self.n_params['n_kappa'],
        n_site = self.n_params['n_site']

        n_kappa_max = np.max(n_kappa)
        self.n_kappa_masks = tf.stack([tf.concat([tf.ones(n_kappa[site]), tf.zeros(n_kappa_max - n_kappa[site])], axis=-1) for site in range(n_site)])

        self.learning_rate = self.n_params['learning_rate']
        self.alpha_1, self.alpha_2 = self.n_params['alpha_1'], self.n_params['alpha_2']

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

            self.gener_x_trans_prec_values =  self.add_weight(name='gener_x_trans_prec_values', shape=(n_x*(n_x+1)//2, ), initializer=tf.constant_initializer(np.hstack([np.ones(n_x), np.zeros(n_x*(n_x+1)//2-n_x)])))


        # Define eta_varialbes in gpu
        self.x_affine_values =  self.add_weight(name='x_affine_values', shape=(n_x*(n_x+1)//2, ), initializer=tf.constant_initializer(np.hstack([np.ones(n_x) * tf.math.log(1 - (1 - EPS) ** 2), np.zeros(n_x*(n_x+1)//2-n_x)])))

        self.recog_kappa_g_mean = self.add_weight(name='recog_kappa_g_mean', shape=(n_x, ), initializer=tf.constant_initializer(np.zeros(n_x)))
        self.recog_kappa_g_prec_values = self.add_weight(name='recog_kappa_g_prec_values', shape=(n_x*(n_x+1)//2, ), initializer=tf.constant_initializer(np.zeros(n_x*(n_x+1)//2)))

        self.recog_y = tf.keras.models.Sequential([tf.keras.layers.Dense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform') for n_layer in n_recog_y])
        self.recog_y_mean, self.recog_y_prec_values = tf.keras.Sequential([tf.keras.layers.Activation('leaky_relu'),  tf.keras.layers.Dense(n_x)]), tf.keras.Sequential([clip_by_value, TriangularDense(n_x)])

        self.recog_kappa = tf.keras.models.Sequential([ParallelDense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform') for n_layer in n_recog_kappa])
        self.recog_kappa_mean, self.recog_kappa_prec_values = ParallelDense(n_x), tf.keras.Sequential([clip_by_value, ParallelTriangularDense(n_x)])

        self.recog = tf.keras.models.Sequential([ParallelDense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform') for n_layer in n_recog])
        self.recog_mean, self.recog_prec_values = ParallelDense(n_z), tf.keras.Sequential([clip_by_value, ParallelTriangularDense(n_z)])

        self.recog_g = tf.keras.models.Sequential([ParallelDense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform') for n_layer in n_recog_g])
        self.recog_g_mean, self.recog_g_prec_values = ParallelDense(n_z), tf.keras.Sequential([clip_by_value, ParallelTriangularDense(n_z)])

        self.gener_x = tf.keras.models.Sequential([ParallelDense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform') for n_layer in n_gener_x])
        self.gener_x_mean, self.gener_x_prec_values = ParallelDense(n_x), tf.keras.Sequential([clip_by_value, ParallelTriangularDense(n_x)])

        self.gener_kappa = tf.keras.models.Sequential([ParallelDense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform') for n_layer in n_gener_kappa])
        self.gener_kappa_mean, self.gener_kappa_prec_values = ParallelDense(n_kappa_max), tf.keras.Sequential([clip_by_value, ParallelDense(n_kappa_max)])

        self.gener_y = tf.keras.models.Sequential([tf.keras.layers.Dense(n_layer, activation='leaky_relu', kernel_initializer='he_uniform') for n_layer in n_gener_y])
        self.gener_y_mean, self.gener_y_prec_values = tf.keras.layers.Dense(n_y), tf.keras.Sequential([clip_by_value, tf.keras.layers.Dense(n_y)])

        self.log_lambda = tf.Variable(tf.zeros(n_site))

        self.recog_y.build((None, None, n_y)), self.recog_y_mean.build((None, None, n_recog_y[-1])), self.recog_y_prec_values.build((None, None, n_recog_y[-1]))
        self.recog_kappa.build((None, n_site, n_kappa_max)), self.recog_kappa_mean.build((None, n_site, n_recog_kappa[-1])), self.recog_kappa_prec_values.build((None, n_site, n_recog_kappa[-1]))

        self.recog.build((None, None, n_site, n_x + n_kappa_max)), self.recog_mean.build((None, None, n_site, n_recog[-1])), self.recog_prec_values.build((None, None, n_site, n_recog[-1]))
        self.recog_g.build((None, None, n_site, n_x)), self.recog_g_mean.build((None, None, n_site, n_recog[-1])), self.recog_g_prec_values.build((None, None, n_site, n_recog[-1]))

        self.gener_x.build((None, None, n_site, n_z)), self.gener_x_mean.build((None, None, n_site, n_gener_x[-1])), self.gener_x_prec_values.build((None, None, n_site, n_gener_x[-1]))
        self.gener_kappa.build((None, None, n_site, n_z)), self.gener_kappa_mean.build((None, None, n_site, n_gener_kappa[-1])), self.gener_kappa_prec_values.build((None, None, n_site, n_gener_kappa[-1]))

        self.gener_y.build((None, None, n_x)), self.gener_y_mean.build((None, None, n_gener_y[-1])), self.gener_y_prec_values.build((None, None, n_gener_y[-1]))

        recog_layers = [self.recog, self.recog_mean, self.recog_prec_values, self.recog_g, self.recog_g_mean, self.recog_g_prec_values]
        gener_layers = [self.gener_x, self.gener_x_mean, self.gener_x_prec_values, self.gener_kappa, self.gener_kappa_mean, self.gener_kappa_prec_values]

        recog_y_layers = [self.recog_y, self.recog_y_mean, self.recog_y_prec_values]
        recog_kappa_layers = [self.recog_kappa, self.recog_kappa_mean, self.recog_kappa_prec_values]

        gener_y_layers = [self.gener_y, self.gener_y_mean, self.gener_y_prec_values]

        self.lambda_recog_variables = sum([layer.variables for layer in recog_layers], [])
        self.lambda_gener_variables = sum([layer.variables for layer in gener_layers], [])
        self.lambda_scale_variables = [self.log_lambda]

        self.eta_variables_3 = sum([layer.variables for layer in gener_y_layers], [])
        with tf.device('/cpu:0'):
            self.eta_variables_2 = [self.recog_x_init_mean, self.recog_x_init_prec_values, self.recog_x_trans_mat_values, self.gener_x_init_mean, self.gener_x_init_prec_values, self.gener_x_trans_mat_values, self.gener_x_trans_mean, self.gener_x_trans_prec_values]
        self.eta_variables_1 = sum([layer.variables for layer in recog_y_layers + recog_kappa_layers], []) + [self.recog_kappa_g_mean, self.recog_kappa_g_prec_values]

        self.lambda_optimizer = tf.keras.optimizers.legacy.Adam(self.learning_rate)
        self.eta_optimizer_3 = tf.keras.optimizers.legacy.Adam(self.learning_rate)
        with tf.device('/cpu:0'):
            self.eta_optimizer_2 = tf.keras.optimizers.legacy.Adam(self.learning_rate)
        self.eta_optimizer_1 = tf.keras.optimizers.legacy.Adam(self.learning_rate)

        self._train, self._eval_log_likelihood, self._eval_log_likelihood_x, self._decode_init, self._decode_update, self._assign_log_lambda = _train, _eval_log_likelihood, _eval_log_likelihood_x, _decode_init, _decode_update, _assign_log_lambda


    def assign_log_lambda(self, batch_kappas, batch_masks, batch_ys, n_monte):
        self._assign_log_lambda(self, batch_kappas, batch_masks, batch_ys, n_monte)


    def train(self, kappas_padded, masks, ys_padded, n_monte):
        return self._train(self, kappas_padded, masks, ys_padded, n_monte)


    def eval_log_likelihood(self, kappas, masks, y, log_scale_train, log_scale_eval, n_monte):
        return self._eval_log_likelihood(self, kappas, masks, y, log_scale_train, log_scale_eval, n_monte)


    def eval_log_likelihood_x(self, kappas, masks, y, log_scale_train, log_scale_eval, n_monte):
        return self._eval_log_likelihood_x(self, kappas, masks, y, log_scale_train, log_scale_eval, n_monte)

    def decode_init(self, kappas, masks, log_scale_train, log_scale_eval, n_monte, n_iter=0.):
        return self._decode_init(self, kappas, masks, log_scale_train, log_scale_eval, n_monte, n_iter)

    def decode_update(self, x_current_means, x_current_covs, kappas, masks, log_scale_train, log_scale_eval, n_monte):
        return self._decode_update(self, x_current_means, x_current_covs, kappas, masks, log_scale_train, log_scale_eval, n_monte)


    def decode(self, kappas, masks, log_scale_train, log_scale_eval, n_monte, n_iter_init, n_iter_update):

        y_smoothed_means, y_smoothed_covs, x_smoothed_means, x_smoothed_covs = self.decode_init(kappas, masks, log_scale_train, log_scale_eval, n_monte, n_iter_init)
        for i in range(n_iter_update):
            y_smoothed_means, y_smoothed_covs, x_smoothed_means, x_smoothed_covs = self.decode_update(x_smoothed_means, x_smoothed_covs, kappas, masks, log_scale_train, log_scale_eval, n_monte)

        return y_smoothed_means, y_smoothed_covs, x_smoothed_means, x_smoothed_covs


def _train(model, batch_kappas, batch_masks, batch_ys, n_monte):

    n_batch, R = tf.shape(batch_ys)[0], tf.shape(batch_ys)[1]
    n_x = model.n_params['n_x']
    n_site = model.n_params['n_site']
    n_compile = model.n_params['n_compile']

    batch_ns = tf.math.count_nonzero(tf.reduce_sum(batch_masks, (2, 3)), axis=-1)
    batch_us = calculateUpperBound(batch_ns, n_compile)

    def _train_batch(state, elem):

        kappas, masks, u, y = elem

        epsx = tf.random.normal((R, 1, n_x), dtype=tf.float32)

        y_not_nan, is_not_nan, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde = compute_eta_loss_1(model, kappas, masks, y)
        with tf.device('/cpu:0'):
            _, x = compute_eta_loss_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx)

        eta_cost_3, x_grad_1 = _apply_eta_gradients_3(model, y_not_nan, is_not_nan, x)

        lambda_recog_cost, lambda_gener_cost, x_grad_2 = _apply_lambda_gradients(model, kappas[:u], masks[:u], x, n_monte)

        x_grad = x_grad_1 + x_grad_2

        with tf.device('/cpu:0'):
            eta_cost_2, x_affine_diag_grad, x_affine_tri_grad, recog_x_prec_tilde_grad, recog_x_mean_dot_prec_tilde_grad = _apply_eta_gradients_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx, x_grad)
        eta_cost_1 = _apply_eta_gradients_1(model, kappas, masks, y, x_affine_diag_grad, x_affine_tri_grad, recog_x_prec_tilde_grad, recog_x_mean_dot_prec_tilde_grad)

        cost = eta_cost_1 + eta_cost_2 + eta_cost_3 + lambda_recog_cost + lambda_gener_cost

        return cost

    costs = tf.scan(_train_batch, elems=(batch_kappas, batch_masks, batch_us, batch_ys), initializer=0.)
    avg_cost = tf.reduce_mean(costs)

    return avg_cost


@tf.function(jit_compile=True, reduce_retracing=True)
def _apply_lambda_gradients(model, kappas, masks, x, n_monte):

    alpha_1, alpha_2 = model.alpha_1, model.alpha_2

    R = tf.shape(x)[0]

    n_site = model.n_params['n_site']

    n_jackknife = model.n_params['n_jackknife']

    comb_masks = [tf.constant(np.eye(n_monte)[np.array(list(combinations(np.arange(n_monte), n_monte-l)))], tf.float32) for l in range(n_jackknife+1)]
    comb_coefs = tf.constant([(- 1.) ** l * (n_monte - l) ** n_jackknife / np.math.factorial(n_jackknife - l) / np.math.factorial(l) for l in range(n_jackknife+1)], tf.float32)

    with tf.GradientTape(persistent=True) as tape:

        tape.watch(x)

        log_lambda = model.log_lambda

        lambda_loss, x_mean, x_prec = _compute_lambda_loss(model, matmul_mask_tranpose_vec(masks, x), kappas, n_monte)
        lambda_g_loss, x_g_mean, x_g_prec = _compute_lambda_g_loss(model, tf.repeat(x, n_site, axis=1), n_monte)

        lambda_recog_weights = (1 - alpha_1) * tf.stop_gradient(tf.exp(2. * tf.math.log_softmax((1 - alpha_1) * lambda_loss, axis=1)))
        lambda_recog_weights += alpha_1 * tf.stop_gradient(tf.exp(tf.math.log_softmax((1 - alpha_1) * lambda_loss, axis=1)))
        lambda_gener_weights = tf.stop_gradient(tf.exp(tf.math.log_softmax((1 - alpha_1) * lambda_loss, axis=1)))

        lambda_recog_all = tf.reduce_sum(tf.math.multiply_no_nan(log_lambda + lambda_loss, lambda_recog_weights), axis=1)
        lambda_gener_all = tf.reduce_sum(tf.math.multiply_no_nan(log_lambda + lambda_loss, lambda_gener_weights), axis=1)

        lambda_g_recog_weights, lambda_g_gener_weights, lambda_g_scale_weights = tf.zeros_like(lambda_g_loss), tf.zeros_like(lambda_g_loss), tf.zeros((R, 1, n_site))

        for l in range(n_jackknife+1):

            lambda_g_loss_resampled = tf.tensordot(lambda_g_loss, comb_masks[l], [[1], [2]])

            lambda_g_recog_weights_resampled = (alpha_2 - 1) * tf.stop_gradient(tf.exp(log_lambda[:, tf.newaxis, tf.newaxis] + alpha_2 ** -1 * tfp.math.reduce_logmeanexp(alpha_2 * lambda_g_loss_resampled, axis=-1, keepdims=True) + 2. * tf.math.log_softmax(alpha_2 * lambda_g_loss_resampled, axis=-1)))
            lambda_g_recog_weights_resampled -= (alpha_2 - 1) * tf.stop_gradient(tf.exp(log_lambda[:, tf.newaxis, tf.newaxis] + alpha_2 ** -1 * tfp.math.reduce_logmeanexp(alpha_2 * lambda_g_loss_resampled, axis=-1, keepdims=True) + tf.math.log_softmax(alpha_2 * lambda_g_loss_resampled, axis=-1)))
            lambda_g_gener_weights_resampled = tf.stop_gradient(tf.exp(log_lambda[:, tf.newaxis, tf.newaxis] + alpha_2 ** -1 * tfp.math.reduce_logmeanexp(alpha_2 * lambda_g_loss_resampled, axis=-1, keepdims=True) + tf.math.log_softmax(alpha_2 * lambda_g_loss_resampled, axis=-1)))
            lambda_g_scale_weights_resampled = tf.stop_gradient(tf.exp(log_lambda[:, tf.newaxis, tf.newaxis] + alpha_2 ** -1 * tfp.math.reduce_logmeanexp(alpha_2 * lambda_g_loss_resampled, axis=-1, keepdims=True)))

            lambda_g_recog_weights += comb_coefs[l] * tf.linalg.matrix_transpose(tf.reduce_mean(tf.matmul(lambda_g_recog_weights_resampled[:, :, :, tf.newaxis], comb_masks[l][tf.newaxis, tf.newaxis]), axis=(2, 3)))
            lambda_g_gener_weights += comb_coefs[l] * tf.linalg.matrix_transpose(tf.reduce_mean(tf.matmul(lambda_g_gener_weights_resampled[:, :, :, tf.newaxis], comb_masks[l][tf.newaxis, tf.newaxis]), axis=(2, 3)))
            lambda_g_scale_weights += comb_coefs[l] * tf.linalg.matrix_transpose(tf.reduce_mean(lambda_g_scale_weights_resampled, axis=2))

        lambda_g_recog_all = tf.reduce_sum(tf.math.multiply_no_nan(log_lambda + lambda_g_loss, lambda_g_recog_weights), axis=1)
        lambda_g_gener_all = tf.reduce_sum(tf.math.multiply_no_nan(log_lambda + lambda_g_loss, lambda_g_gener_weights), axis=1)
        lambda_g_scale_all = tf.reduce_sum(tf.math.multiply_no_nan(log_lambda, lambda_g_scale_weights), axis=1)

        lambda_recog_cost = - tf.reduce_sum(lambda_recog_all * tf.reduce_sum(masks, axis=1)) + tf.reduce_sum(lambda_g_recog_all)
        lambda_gener_cost = - tf.reduce_sum(lambda_gener_all * tf.reduce_sum(masks, axis=1)) + tf.reduce_sum(lambda_g_gener_all)
        lambda_scale_cost = - tf.reduce_sum(log_lambda * tf.reduce_sum(masks, axis=1)) + tf.reduce_sum(lambda_g_scale_all)

    lambda_recog_gradients = tape.gradient(lambda_recog_cost, model.lambda_recog_variables)
    lambda_gener_gradients = tape.gradient(lambda_gener_cost, model.lambda_gener_variables)
    lambda_scale_gradients = tape.gradient(lambda_scale_cost, model.lambda_scale_variables)

    vec = - tf.linalg.matvec(x_prec, matmul_mask_tranpose_vec(masks, x)[:, tf.newaxis] - x_mean)
    x_gener_grad = tf.reduce_sum(tf.math.multiply_no_nan(vec, lambda_gener_weights[:, :, :, tf.newaxis]), axis=1)

    vec_g = - tf.linalg.matvec(x_g_prec, x[:, tf.newaxis] - x_g_mean)
    x_g_gener_grad = tf.reduce_sum(tf.math.multiply_no_nan(vec_g, lambda_g_gener_weights[:, :, :, tf.newaxis]), axis=1)

    x_grad = tf.reduce_sum(- matmul_mask_vec(masks, x_gener_grad) + x_g_gener_grad, axis=1, keepdims=True)
    x_grad += tape.gradient(lambda_recog_cost, x)

    if not tf.math.reduce_any([tf.math.reduce_any([tf.math.is_nan(grad), tf.math.is_inf(grad)]) for grad in lambda_recog_gradients + lambda_gener_gradients + lambda_scale_gradients if grad is not None]):

        model.lambda_optimizer.apply_gradients(zip(lambda_recog_gradients, model.lambda_recog_variables))
        model.lambda_optimizer.apply_gradients(zip(lambda_gener_gradients, model.lambda_gener_variables))
        model.lambda_optimizer.apply_gradients(zip(lambda_scale_gradients, model.lambda_scale_variables))

    return lambda_recog_cost, lambda_gener_cost, x_grad


def _compute_lambda_loss(model, x, kappas, n_monte):

    n = tf.shape(x)[0]
    n_z, n_x = model.n_params['n_z'], model.n_params['n_x']
    n_site = model.n_params['n_site']

    n_kappa_max = tf.reduce_max(model.n_params['n_kappa'])

    rz = model.recog(tf.concat([x, kappas], axis=-1)[:, tf.newaxis])
    z_mean, z_prec_values = model.recog_mean(rz), model.recog_prec_values(rz)
    z_prec_diag, z_prec_tri = z_prec_values[:, :, :, :n_z], fill_triangular(z_prec_values[:, :, :, n_z:])

    eps = tf.random.normal((n, n_monte, n_site, n_z))
    z = z_mean + tf.squeeze(tf.linalg.triangular_solve(tf.linalg.matrix_transpose(z_prec_tri), (eps * tf.exp(- 0.5 * z_prec_diag))[:, :, :, :, tf.newaxis], lower=False), axis=-1)

    gz_x = model.gener_x(z)
    x_mean, x_prec_values = model.gener_x_mean(gz_x), model.gener_x_prec_values(gz_x)
    x_prec_diag, x_prec_tri = x_prec_values[:, :, :, :n_x], fill_triangular(x_prec_values[:, :, :, n_x:])

    gz_kappa = model.gener_kappa(z)
    kappa_mean, kappa_prec_diag = model.gener_kappa_mean(gz_kappa), model.gener_kappa_prec_values(gz_kappa)

    reconstr_x_loss = - 0.5 * n_x * tf.math.log(2 * np.pi) + 0.5 * tf.reduce_sum(x_prec_diag, -1) - 0.5 * tf.reduce_sum(tf.square(tf.linalg.matvec(tf.linalg.matrix_transpose(x_prec_tri), (tf.stop_gradient(x)[:, tf.newaxis] - x_mean)) * tf.exp(0.5 * x_prec_diag)), -1)
    reconstr_kappa_loss = tf.reduce_sum(tf.math.multiply_no_nan(- 0.5 * tf.math.log(2 * np.pi) + 0.5 * kappa_prec_diag - 0.5 * tf.square((kappas[:, tf.newaxis] - kappa_mean) * tf.exp(0.5 * kappa_prec_diag)), model.n_kappa_masks), -1)
    latent_loss = - 0.5 * tf.reduce_sum(tf.square(z), -1) - 0.5 * tf.reduce_sum(tf.stop_gradient(z_prec_diag), -1) + 0.5 * tf.reduce_sum(tf.square(tf.linalg.matvec(tf.linalg.matrix_transpose(tf.stop_gradient(z_prec_tri)), z - tf.stop_gradient(z_mean)) * tf.exp(0.5 * tf.stop_gradient(z_prec_diag))), -1)

    lambda_loss = reconstr_x_loss + reconstr_kappa_loss + latent_loss

    x_prec = tf.matmul(x_prec_tri * tf.exp(0.5 * x_prec_diag)[:, :, :, tf.newaxis], tf.linalg.matrix_transpose(x_prec_tri * tf.exp(0.5 * x_prec_diag)[:, :, :, tf.newaxis]))

    return lambda_loss, x_mean, x_prec


def _compute_lambda_g_loss(model, x, n_monte):

    n = tf.shape(x)[0]
    n_z, n_x = model.n_params['n_z'], model.n_params['n_x']
    n_site = model.n_params['n_site']

    rz = model.recog_g(x[:, tf.newaxis])
    z_mean, z_prec_values = model.recog_g_mean(rz), model.recog_g_prec_values(rz)
    z_prec_diag, z_prec_tri = z_prec_values[:, :, :, :n_z], fill_triangular(z_prec_values[:, :, :, n_z:])

    eps = tf.random.normal((n, n_monte, n_site, n_z))
    z = z_mean + tf.squeeze(tf.linalg.triangular_solve(tf.linalg.matrix_transpose(z_prec_tri), (eps * tf.exp(- 0.5 * z_prec_diag))[:, :, :, :, tf.newaxis], lower=False), axis=-1)

    gz_x = model.gener_x(z)
    x_mean, x_prec_values = model.gener_x_mean(gz_x), model.gener_x_prec_values(gz_x)
    x_prec_diag, x_prec_tri = x_prec_values[:, :, :, :n_x], fill_triangular(x_prec_values[:, :, :, n_x:])

    reconstr_x_loss = - 0.5 * n_x * tf.math.log(2 * np.pi) + 0.5 * tf.reduce_sum(x_prec_diag, -1) - 0.5 * tf.reduce_sum(tf.square(tf.linalg.matvec(tf.linalg.matrix_transpose(x_prec_tri), (tf.stop_gradient(x)[:, tf.newaxis] - x_mean)) * tf.exp(0.5 * x_prec_diag)), -1)
    latent_loss = - 0.5 * tf.reduce_sum(tf.square(z), -1) - 0.5 * tf.reduce_sum(tf.stop_gradient(z_prec_diag), -1) + 0.5 * tf.reduce_sum(tf.square(tf.linalg.matvec(tf.linalg.matrix_transpose(tf.stop_gradient(z_prec_tri)), z - tf.stop_gradient(z_mean)) * tf.exp(0.5 * tf.stop_gradient(z_prec_diag))), -1)

    lambda_g_loss = reconstr_x_loss + latent_loss

    x_prec = tf.matmul(x_prec_tri * tf.exp(0.5 * x_prec_diag)[:, :, :, tf.newaxis], tf.linalg.matrix_transpose(x_prec_tri * tf.exp(0.5 * x_prec_diag)[:, :, :, tf.newaxis]))

    return lambda_g_loss, x_mean, x_prec


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

compute_lambda_loss = tf.function(_compute_lambda_loss, jit_compile=True)
compute_lambda_g_loss = tf.function(_compute_lambda_g_loss, jit_compile=True)



def compute_lambda_loss_split(model, x, kappas, n_monte, split_num=1):

    split_size = tf.shape(kappas)[0] // split_num + 1

    lambda_loss = []

    for i in range(split_num):
        x_splited, kappas_splited = x[i*split_size:(i+1)*split_size], kappas[i*split_size:(i+1)*split_size]
        lambda_loss_splited, _, _ = compute_lambda_loss(model, x_splited, kappas_splited, n_monte)
        lambda_loss.append(lambda_loss_splited)

    lambda_loss = tf.concat(lambda_loss, axis=0)

    return lambda_loss


def compute_lambda_g_loss_split(model, x, n_monte, split_num=1):

    split_size = tf.shape(x)[0] // split_num + 1

    lambda_g_loss = []

    for i in range(split_num):
        x_splited = x[i*split_size:(i+1)*split_size]
        lambda_g_loss_splited, _, _ = compute_lambda_g_loss(model, x_splited, n_monte)
        lambda_g_loss.append(lambda_g_loss_splited)

    lambda_g_loss = tf.concat(lambda_g_loss, axis=0)

    return lambda_g_loss



@tf.function
def _eval_log_likelihood(model, kappas, masks, y, log_scale_train, log_scale_eval, n_monte):

    R = tf.shape(y)[0]
    n_x = model.n_params['n_x']
    n_site = model.n_params['n_site']

    alpha_1, alpha_2 = model.alpha_1, model.alpha_2

    epsx = tf.random.normal((R, 1, n_x), dtype=tf.float32)

    y_not_nan, is_not_nan, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde = compute_eta_loss_1(model, kappas, masks, y)
    with tf.device('/cpu:0'):
        eta_cost_2, x = compute_eta_loss_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx)
    eta_cost_3 = compute_eta_loss_3(model, y_not_nan, is_not_nan, x)

    log_likelihood = - eta_cost_2 - eta_cost_3

    log_lambda = model.log_lambda

    lambda_loss = compute_lambda_loss_split(model, matmul_mask_tranpose_vec(masks, x), kappas, n_monte, n_site)
    lambda_g_loss = compute_lambda_g_loss_split(model, tf.repeat(x, n_site, axis=1), n_monte, n_site)

    lambda_scale_all = log_scale_train + (1 - alpha_1) ** -1 * tfp.math.reduce_logmeanexp((1 - alpha_1) * (log_lambda + lambda_loss), axis=1)
    lambda_g_scale_all = tf.exp(log_scale_train + alpha_2 ** -1 * tfp.math.reduce_logmeanexp(alpha_2 * (log_lambda + lambda_g_loss), axis=1) - log_scale_eval)

    log_likelihood += tf.reduce_sum(tf.math.multiply_no_nan(tf.reduce_sum(masks, axis=1), lambda_scale_all)) - tf.reduce_sum(lambda_g_scale_all)


    return log_likelihood


@tf.function
def _eval_log_likelihood_x(model, kappas, masks, y, log_scale_train, log_scale_eval, n_monte):

    R = tf.shape(y)[0]
    n_x = model.n_params['n_x']
    n_site = model.n_params['n_site']

    alpha_1, alpha_2 = model.alpha_1, model.alpha_2

    epsx = tf.random.normal((R, 1, n_x), dtype=tf.float32)

    y_not_nan, is_not_nan, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde = compute_eta_loss_1(model, kappas, masks, y)
    with tf.device('/cpu:0'):
        eta_cost_2, x = compute_eta_loss_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx)
    eta_cost_3 = compute_eta_loss_3(model, y_not_nan, is_not_nan, x)

    log_likelihood_x = - eta_cost_2 - eta_cost_3

    log_lambda = model.log_lambda

    lambda_loss = compute_lambda_g_loss_split(model, matmul_mask_tranpose_vec(masks, x), n_monte, n_site)
    lambda_g_loss = compute_lambda_g_loss_split(model, tf.repeat(x, n_site, axis=1), n_monte, n_site)

    lambda_scale_all = log_scale_train + (1 - alpha_1) ** -1 * tfp.math.reduce_logmeanexp((1 - alpha_1) * (log_lambda + lambda_loss), axis=1)
    lambda_g_scale_all = tf.exp(log_scale_train + alpha_2 ** -1 * tfp.math.reduce_logmeanexp(alpha_2 * (log_lambda + lambda_g_loss), axis=1) - log_scale_eval)

    log_likelihood_x += tf.reduce_sum(tf.math.multiply_no_nan(tf.reduce_sum(masks, axis=1), lambda_scale_all)) - tf.reduce_sum(lambda_g_scale_all)

    return log_likelihood_x


def _eval_log_lambda_derivative(model, x, kappas, log_scale_train, n_monte):

    alpha_1 = model.alpha_1

    n_x = model.n_params['n_x']

    lambda_loss, x_mean, x_prec = _compute_lambda_loss(model, x, kappas, n_monte)
    weights = tf.math.softmax((1 - alpha_1) * lambda_loss, axis=1)
    vec = - tf.linalg.matvec(x_prec, x[:, tf.newaxis] - x_mean)

    weighted_vec = tf.reduce_sum(weights[:, :, :, tf.newaxis] * vec, axis=1)

    grad_log_lambda_x = weighted_vec
    hessian_log_lambda_x = - weighted_vec[:, :, tf.newaxis] * weighted_vec[:, :, :, tf.newaxis] - tf.reduce_sum(alpha_1 * (vec - weighted_vec[:, tf.newaxis])[:, :, :, tf.newaxis] * (vec - weighted_vec[:, tf.newaxis])[:, :, :, :, tf.newaxis] * weights[:, :, :, tf.newaxis, tf.newaxis], axis=1)

    return grad_log_lambda_x, hessian_log_lambda_x


def _eval_lambda_g_derivative(model, x, log_scale_train, n_monte):

    alpha_2 = model.alpha_2

    n_x = model.n_params['n_x']
    n_site = model.n_params['n_site']

    log_lambda = model.log_lambda

    lambda_g_loss, x_mean, x_prec = _compute_lambda_g_loss(model, x, n_monte)

    vec = - tf.linalg.matvec(x_prec, x[:, tf.newaxis] - x_mean)

    weights = tf.math.softmax(alpha_2 * lambda_g_loss, axis=1)
    scale = tf.exp(log_scale_train + log_lambda + alpha_2 ** -1 * tfp.math.reduce_logmeanexp(alpha_2 * lambda_g_loss, axis=1))
    weighted_vec = tf.reduce_sum(weights[:, :, :, tf.newaxis] * vec, axis=1)

    grad_lambda_x = scale[:, :, tf.newaxis] * weighted_vec
    hessian_lambda_x = scale[:, :, tf.newaxis, tf.newaxis] * (weighted_vec[:, :, tf.newaxis] * weighted_vec[:, :, :, tf.newaxis] + tf.reduce_sum(alpha_2 * (vec - weighted_vec[:, tf.newaxis])[:, :, :, tf.newaxis] * (vec - weighted_vec[:, tf.newaxis])[:, :, :, :, tf.newaxis] * weights[:, :, :, tf.newaxis, tf.newaxis], axis=1))

    return grad_lambda_x, hessian_lambda_x


@tf.function(experimental_compile=True)
def eval_log_lambda_derivative(model, x, kappas, log_scale_train, n_monte):
    return _eval_log_lambda_derivative(model, x, kappas, log_scale_train, n_monte)


@tf.function(experimental_compile=True)
def eval_lambda_g_derivative(model, x, log_scale_train, n_monte):
    return _eval_lambda_g_derivative(model, x, log_scale_train, n_monte)


def eval_log_lambda_derivative_split(model, x, kappas, log_scale_train, n_monte, split_num=1):

    split_size = tf.shape(kappas)[0] // split_num + 1

    grad_log_lambda_x, hessian_log_lambda_x = [], []

    for i in range(split_num):
        x_splited, kappas_splited = x[i*split_size:(i+1)*split_size], kappas[i*split_size:(i+1)*split_size]
        grad_log_lambda_x_splited, hessian_log_lambda_x_splited = eval_log_lambda_derivative(model, x_splited, kappas_splited, log_scale_train, n_monte)
        grad_log_lambda_x.append(grad_log_lambda_x_splited), hessian_log_lambda_x.append(hessian_log_lambda_x_splited)

    grad_log_lambda_x, hessian_log_lambda_x = tf.concat(grad_log_lambda_x, axis=0), tf.concat(hessian_log_lambda_x, axis=0)

    return grad_log_lambda_x, hessian_log_lambda_x


def eval_lambda_g_derivative_split(model, x, log_scale_train, n_monte, split_num=1):

    split_size = tf.shape(x)[0] // split_num + 1

    grad_lambda_x, hessian_lambda_x = [], []

    for i in range(split_num):
        x_splited = x[i*split_size:(i+1)*split_size]
        grad_lambda_x_splited, hessian_lambda_x_splited = eval_lambda_g_derivative(model, x_splited, log_scale_train, n_monte)
        grad_lambda_x.append(grad_lambda_x_splited), hessian_lambda_x.append(hessian_lambda_x_splited)

    grad_lambda_x, hessian_lambda_x = tf.concat(grad_lambda_x, axis=0), tf.concat(hessian_lambda_x, axis=0)

    return grad_lambda_x, hessian_lambda_x


@tf.function
def _decode_init(model, kappas, masks, log_scale_train, log_scale_eval, n_monte, n_iter):

    n_site = model.n_params['n_site']
    n_kappa_max = tf.reduce_max(model.n_params['n_kappa'])
    R = tf.shape(masks)[1]

    inds = [tf.boolean_mask(tf.repeat(tf.range(R)[tf.newaxis], tf.shape(masks[:, :, site])[0], axis=0), masks[:, :, site]) for site in range(n_site)]
    n_max = tf.reduce_max([tf.reduce_max(tf.unique_with_counts(inds[site])[-1]) for site in range(n_site)])

    kappas_ragged = tf.stack([tf.RaggedTensor.from_value_rowids(tf.boolean_mask(tf.squeeze(kappas[:, site]), tf.reduce_sum(masks[:, :, site], axis=1)), inds[site], R).to_tensor(shape=(R, n_max, n_kappa_max)) for site in range(n_site)], axis=2)
    masks_ragged = tf.stack([tf.RaggedTensor.from_value_rowids(tf.ones_like(inds[site], dtype=tf.float32), inds[site], R).to_tensor(shape=(R, n_max)) for site in range(n_site)], axis=2)[:, :, tf.newaxis]

    n_x, n_y = model.n_params['n_x'], model.n_params['n_y']

    kappas_ragged = tf.repeat(kappas_ragged, n_iter, axis=0)
    masks_ragged = tf.repeat(masks_ragged, n_iter, axis=0)
    updates = tf.reshape(tf.concat([tf.zeros((R, n_iter-1)), tf.ones((R, 1))], axis=1), (R*n_iter, ))

    x_filtered_state = collections.namedtuple('x_filtered_state', ['x_filtered_mean', 'x_filtered_prec', 'x_predicted_mean', 'x_predicted_prec'])
    x_smoothed_state = collections.namedtuple('x_smoothed_state', ['x_smoothed_mean', 'x_smoothed_cov'])

    x_affine_diag, x_affine_tri = model.x_affine_values[:n_x], fill_triangular(model.x_affine_values[n_x:])

    x_init_prec_diag, x_init_prec_tri = model.gener_x_init_prec_values[:n_x], fill_triangular(model.gener_x_init_prec_values[n_x:])
    x_init_prec = tf.matmul(x_init_prec_tri * tf.exp(0.5 * x_init_prec_diag), tf.transpose(x_init_prec_tri * tf.exp(0.5 * x_init_prec_diag)))
    x_init_mean = tf.reshape(model.gener_x_init_mean, (1, n_x))

    x_trans_mean = tf.reshape(model.gener_x_trans_mean, (1, n_x))
    x_trans_mat = tf.reshape(model.gener_x_trans_mat_values, [n_x, n_x])
    x_trans_mat /= tf.nn.relu(tf.reduce_max(tf.abs(tf.linalg.eigvals(x_trans_mat))) - 1.) + 1.

    x_trans_prec_diag, x_trans_prec_tri = model.gener_x_trans_prec_values[:n_x], fill_triangular(model.gener_x_trans_prec_values[n_x:])
    x_trans_prec = tf.matmul(x_trans_prec_tri * tf.exp(0.5 * x_trans_prec_diag), tf.transpose(x_trans_prec_tri * tf.exp(0.5 * x_trans_prec_diag)))

    def _filter_predict(model, x_filtered_mean, x_filtered_prec):
        x_predicted_mean, x_predicted_prec = x_trans_mean + tf.matmul(x_filtered_mean, x_trans_mat), tf.linalg.inv(tf.matmul(tf.transpose(x_trans_mat), tf.linalg.solve(x_filtered_prec, x_trans_mat)) + tf.linalg.inv(x_trans_prec))
        return x_predicted_mean, x_predicted_prec

    def _filter_correct(model, kappa, mask, x_current_mean, x_predicted_mean, x_predicted_prec, log_scale_train, log_scale_eval, n_monte):

        x_current_mean_orig = tf.transpose(tf.matmul(tf.transpose(x_affine_tri), tf.transpose(x_current_mean * tf.exp(0.5 * x_affine_diag))))

        grad_log_lambda_x, hessian_log_lambda_x = eval_log_lambda_derivative(model, matmul_mask_tranpose_vec(mask, x_current_mean_orig[tf.newaxis]), kappa, log_scale_train, n_monte)
        grad_lambda_x, hessian_lambda_x = eval_lambda_g_derivative(model, tf.repeat(x_current_mean_orig[tf.newaxis], n_site, axis=1), log_scale_train, n_monte)

        likelihood_grads = tf.squeeze(tf.reduce_sum(- matmul_mask_vec(mask, grad_log_lambda_x) + grad_lambda_x * tf.exp(- log_scale_eval), axis=1, keepdims=True), axis=0)
        likelihood_hessians = tf.squeeze(tf.reduce_sum(- matmul_mask_mat(mask, hessian_log_lambda_x) + hessian_lambda_x * tf.exp(- log_scale_eval), axis=1), axis=0)

        likelihood_grads_tilde = tf.linalg.matrix_transpose(tf.matmul(x_affine_tri, tf.linalg.matrix_transpose(likelihood_grads))) * tf.exp(0.5 * x_affine_diag)
        likelihood_hessians_tilde = tf.exp(0.5 * x_affine_diag[:, tf.newaxis] + 0.5 * x_affine_diag) * tf.matmul(x_affine_tri, tf.linalg.matrix_transpose(tf.matmul(x_affine_tri, likelihood_hessians)))

        x_filtered_prec = x_predicted_prec + likelihood_hessians_tilde
        x_filtered_mean = x_current_mean + tf.transpose(tf.linalg.solve(x_predicted_prec + likelihood_hessians_tilde, tf.transpose(- likelihood_grads_tilde - tf.linalg.matmul(x_current_mean - x_predicted_mean, x_predicted_prec))))

        return x_filtered_mean, x_filtered_prec


    def _smooth_update(model, x_predicted_mean, x_predicted_prec, x_filtered_mean, x_filtered_prec, x_next_smoothed_mean, x_next_smoothed_cov):

        n_x = model.n_params['n_x']

        x_kalman_gain = tf.matmul(tf.linalg.solve(x_filtered_prec, x_trans_mat), x_predicted_prec)
        x_smoothed_mean = x_filtered_mean + tf.matmul(x_next_smoothed_mean - x_predicted_mean, tf.transpose(x_kalman_gain))
        x_smoothed_cov = tf.linalg.inv(x_filtered_prec + tf.matmul(tf.matmul(x_trans_mat, x_trans_prec), tf.transpose(x_trans_mat))) + tf.matmul(tf.matmul(x_kalman_gain, x_next_smoothed_cov), tf.transpose(x_kalman_gain))

        return x_smoothed_mean, x_smoothed_cov


    def update_forward_fn(state, elem):
        kappa, mask, update = elem

        x_filtered_mean, x_filtered_prec = _filter_correct(model, kappa, mask, state.x_filtered_mean, state.x_predicted_mean, state.x_predicted_prec, log_scale_train, log_scale_eval, n_monte)
        x_predicted_mean, x_predicted_prec = _filter_predict(model, x_filtered_mean, x_filtered_prec)

        return x_filtered_state(x_filtered_mean, x_filtered_prec, x_predicted_mean * update + state.x_predicted_mean * (1 - update), x_predicted_prec * update + state.x_predicted_prec * (1 - update))


    def update_backward_fn(state, elem):

        x_filtered_mean, x_filtered_prec, x_predicted_mean, x_predicted_prec = elem
        x_smoothed_mean, x_smoothed_cov = _smooth_update(model, x_predicted_mean, x_predicted_prec, x_filtered_mean, x_filtered_prec, state.x_smoothed_mean, state.x_smoothed_cov)

        return x_smoothed_state(x_smoothed_mean, x_smoothed_cov)

    x_filtered_means, x_filtered_precs, x_predicted_means, x_predicted_precs = tf.scan(update_forward_fn, elems=(kappas_ragged, masks_ragged, updates), initializer=x_filtered_state(x_init_mean, x_init_prec, x_init_mean, x_init_prec))
    x_filtered_means, x_filtered_precs, x_predicted_means, x_predicted_precs = x_filtered_means[n_iter-1::n_iter], x_filtered_precs[n_iter-1::n_iter], x_predicted_means[n_iter-1::n_iter], x_predicted_precs[n_iter-1::n_iter]

    x_smoothed_mean, x_smoothed_cov = x_filtered_means[-1], tf.linalg.inv(x_filtered_precs[-1])

    x_smoothed_means, x_smoothed_covs = tf.scan(update_backward_fn, elems=(x_filtered_means[:-1], x_filtered_precs[:-1], x_predicted_means[:-1], x_predicted_precs[:-1]), initializer=x_smoothed_state(x_smoothed_mean, x_smoothed_cov), reverse=True)
    x_smoothed_means, x_smoothed_covs = tf.concat([x_smoothed_means, x_smoothed_mean[tf.newaxis]], axis=0), tf.concat([x_smoothed_covs, x_smoothed_cov[tf.newaxis]], axis=0)
    x_smoothed_covs_cholesky = tf.linalg.cholesky(x_smoothed_covs)

    x_smoothed_means = tf.linalg.matrix_transpose(tf.matmul(tf.transpose(x_affine_tri), tf.linalg.matrix_transpose(x_smoothed_means * tf.exp(0.5 * x_affine_diag))))
    x_smoothed_covs_cholesky = tf.matmul(tf.transpose(x_affine_tri), x_smoothed_covs_cholesky * tf.exp(0.5 * x_affine_diag[:, tf.newaxis]))
    x_smoothed_covs = tf.matmul(x_smoothed_covs_cholesky, tf.linalg.matrix_transpose(x_smoothed_covs_cholesky))

    epsx = tf.random.normal((R, n_monte, n_x))
    x = x_smoothed_means + tf.matmul(epsx, tf.linalg.matrix_transpose(x_smoothed_covs_cholesky))

    gy = model.gener_y(x)
    y_mean, y_prec_diag = model.gener_y_mean(gy), model.gener_y_prec_values(gy)

    y_smoothed_means = tf.reduce_mean(y_mean, 1, keepdims=True)
    y_smoothed_covs = tf.reduce_mean(tf.matmul((y_smoothed_means - y_mean)[:, :, :, tf.newaxis], (y_smoothed_means - y_mean)[:, :, tf.newaxis]), 1) + tf.reduce_mean(tf.linalg.diag(tf.exp(- y_prec_diag)), axis=1)

    return y_smoothed_means, y_smoothed_covs, x_smoothed_means, x_smoothed_covs



@tf.function
def _decode_update(model, x_current_means, x_current_covs, kappas, masks, log_scale_train, log_scale_eval, n_monte):

    R = tf.shape(x_current_means)[0]

    n_site = model.n_params['n_site']
    n_x = model.n_params['n_x']
    n_y = model.n_params['n_y']

    x_affine_diag, x_affine_tri = model.x_affine_values[:n_x], fill_triangular(model.x_affine_values[n_x:])

    x_init_prec_diag, x_init_prec_tri = model.gener_x_init_prec_values[:n_x], fill_triangular(model.gener_x_init_prec_values[n_x:])
    x_init_prec = tf.matmul(x_init_prec_tri * tf.exp(0.5 * x_init_prec_diag), tf.transpose(x_init_prec_tri * tf.exp(0.5 * x_init_prec_diag)))
    x_init_mean = tf.reshape(model.gener_x_init_mean, (1, n_x))

    x_trans_prec_diag, x_trans_prec_tri = model.gener_x_trans_prec_values[:n_x], fill_triangular(model.gener_x_trans_prec_values[n_x:])
    x_trans_prec = tf.matmul(x_trans_prec_tri * tf.exp(0.5 * x_trans_prec_diag), tf.transpose(x_trans_prec_tri * tf.exp(0.5 * x_trans_prec_diag)))
    x_trans_mean = tf.reshape(model.gener_x_trans_mean, (1, n_x))
    x_trans_mat = tf.reshape(model.gener_x_trans_mat_values, [n_x, n_x])
    x_trans_mat /= tf.nn.relu(tf.reduce_max(tf.abs(tf.linalg.eigvals(x_trans_mat))) - 1.) + 1.

    grad_log_lambda_x, hessian_log_lambda_x = eval_log_lambda_derivative_split(model, matmul_mask_tranpose_vec(masks, x_current_means), kappas, log_scale_train, n_monte, n_site)
    grad_lambda_x, hessian_lambda_x = eval_lambda_g_derivative_split(model, tf.repeat(x_current_means, n_site, axis=1), log_scale_train, n_monte, n_site)

    likelihood_grads = tf.reduce_sum(- matmul_mask_vec(masks, grad_log_lambda_x) + grad_lambda_x * tf.exp(- log_scale_eval), axis=1, keepdims=True)
    likelihood_hessians = tf.reduce_sum(- matmul_mask_mat(masks, hessian_log_lambda_x) + hessian_lambda_x * tf.exp(- log_scale_eval), axis=1)

    likelihood_grads_tilde = tf.linalg.matrix_transpose(tf.matmul(x_affine_tri, tf.linalg.matrix_transpose(likelihood_grads))) * tf.exp(0.5 * x_affine_diag)
    likelihood_hessians_tilde = tf.exp(0.5 * x_affine_diag[:, tf.newaxis] + 0.5 * x_affine_diag) * tf.linalg.matmul(x_affine_tri, tf.linalg.matrix_transpose(tf.matmul(x_affine_tri, likelihood_hessians)))

    x_current_means_tilde = tf.linalg.matrix_transpose(tf.linalg.triangular_solve(tf.transpose(x_affine_tri), tf.linalg.matrix_transpose(x_current_means), lower=False)) * tf.exp(- 0.5 * x_affine_diag)
    y_tilde = tf.concat([tf.matmul(x_current_means_tilde[:1] - x_init_mean, x_init_prec), tf.matmul(x_current_means_tilde[1:] - tf.matmul(x_current_means_tilde[:-1], x_trans_mat) - x_trans_mean, x_trans_prec)], axis=0)
    gener_x_grads = - likelihood_grads_tilde - tf.concat([y_tilde[:-1] - tf.matmul(y_tilde[1:], tf.transpose(x_trans_mat)), y_tilde[-1:]], axis=0)

    epsx = tf.random.normal((R, n_monte, n_x))

    with tf.device('/cpu:0'):
        x_cholesky_diags, x_cholesky_off_diags, vs_tilde, ws_tilde = cholesky_update(likelihood_hessians_tilde, gener_x_grads, x_trans_mat, x_trans_prec, x_init_prec, epsx)
        x_next_covs_tilde, x_next_pair_covs_tilde = covariance_update(x_cholesky_diags, x_cholesky_off_diags)

    vs = tf.linalg.matrix_transpose(tf.matmul(tf.transpose(x_affine_tri), tf.linalg.matrix_transpose(vs_tilde * tf.exp(0.5 * x_affine_diag))))
    ws = tf.linalg.matrix_transpose(tf.matmul(tf.transpose(x_affine_tri), tf.linalg.matrix_transpose(ws_tilde * tf.exp(0.5 * x_affine_diag))))

    x_next_means = x_current_means + vs
    x_next_covs = tf.linalg.matrix_transpose(tf.matmul(tf.transpose(x_affine_tri), tf.linalg.matrix_transpose(tf.matmul(tf.transpose(x_affine_tri), x_next_covs_tilde * tf.exp(0.5 * x_affine_diag[:, tf.newaxis] + 0.5 * x_affine_diag)))))

    x = x_next_means + ws

    gy = model.gener_y(x)
    y_mean, y_prec_diag = model.gener_y_mean(gy), model.gener_y_prec_values(gy)

    y_smoothed_means = tf.reduce_mean(y_mean, 1, keepdims=True)
    y_smoothed_covs = tf.reduce_mean(tf.matmul((y_smoothed_means - y_mean)[:, :, :, tf.newaxis], (y_smoothed_means - y_mean)[:, :, tf.newaxis]), 1) + tf.reduce_mean(tf.linalg.diag(tf.exp(- y_prec_diag)), axis=1)

    return y_smoothed_means, y_smoothed_covs, x_next_means, x_next_covs


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



def _cholesky_update_gpu(x_hessian_diags, x_grads, x_trans_mat, x_trans_prec, x_init_prec, epsx):

    diags = tf.concat([x_hessian_diags[:1] + x_init_prec + tf.matmul(tf.matmul(x_trans_mat, x_trans_prec), tf.transpose(x_trans_mat)), x_hessian_diags[1:-1] + x_trans_prec + tf.matmul(tf.matmul(x_trans_mat, x_trans_prec), tf.transpose(x_trans_mat)), x_hessian_diags[-1:] + x_trans_prec], axis=0)
    off_diag = - tf.matmul(x_trans_prec, tf.transpose(x_trans_mat))

    mat_diag = tf.reshape(tf.transpose(tf.eye(R)[:, :, tf.newaxis, tf.newaxis] * tf.repeat(diags[:, tf.newaxis], R, axis=1), (0, 2, 1, 3)), (R*n_x, R*n_x))
    mat_off_diag = tf.reshape(tf.transpose(tf.eye(R+1)[:-1, 1:][:, :, tf.newaxis, tf.newaxis] * off_diag, (0, 2, 1, 3)), (R*n_x, R*n_x))

    mat = mat_diag + mat_off_diag + tf.transpose(mat_off_diag)
    chol = tf.linalg.cholesky(mat)

    chol_block = tf.transpose(tf.reshape(chol, (R, n_x, R, n_x)), (0, 2, 1, 3))
    x_cholesky_diags = tf.transpose(tf.linalg.diag_part(tf.transpose(chol_block)))
    x_cholesky_off_diags = tf.concat([tf.transpose(tf.linalg.diag_part(tf.transpose(chol_block[1:, :-1]))), tf.zeros((1, n_x, n_x))], axis=0)

    vs = tf.reshape(tf.linalg.cholesky_solve(chol, tf.reshape(x_grads, (-1, 1))), (R, 1, n_x))
    ws = tf.transpose(tf.reshape(tf.linalg.triangular_solve(tf.transpose(chol), tf.reshape(tf.transpose(epsx, (0, 2, 1)), (-1, n_monte)), lower=False), (R, n_x, n_monte)), (0, 2, 1))

    return x_cholesky_diags, x_cholesky_off_diags, vs, ws



@tf.function(experimental_compile=True)
def cholesky_update(x_hessian_diags, x_grads, x_trans_mat, x_trans_prec, x_init_prec, epsx):
    return _cholesky_update(x_hessian_diags, x_grads, x_trans_mat, x_trans_prec, x_init_prec, epsx)


@tf.function(experimental_compile=True)
def cholesky_update_gpu(x_hessian_diags, x_grads, x_trans_mat, x_trans_prec, x_init_prec, epsx):
    return _cholesky_update_gpu(x_hessian_diags, x_grads, x_trans_mat, x_trans_prec, x_init_prec, epsx)


def _covariance_update(x_cholesky_diags, x_cholesky_off_diags):

    n_x = tf.shape(x_cholesky_diags)[-1]

    x_backward_state = collections.namedtuple('x_backward_state', ['x_smoothed_cov', 'x_smoothed_pair_cov'])
    x_backward_init_state = x_backward_state(tf.zeros((n_x, n_x)), tf.zeros((n_x, n_x)))

    x_cholesky_inv_diags = tf.linalg.triangular_solve(x_cholesky_diags, tf.eye(n_x) + tf.zeros_like(x_cholesky_diags))

    def update_backward_fn(state, elem):

        x_cholesky_inv_diag, x_cholesky_off_diag = elem
        x_smoothed_cov, x_smoothed_pair_cov = state.x_smoothed_cov, state.x_smoothed_pair_cov

        x_smoothed_pair_cov = - tf.matmul(tf.matmul(x_smoothed_cov, x_cholesky_off_diag), x_cholesky_inv_diag)
        x_smoothed_cov = tf.matmul(tf.transpose(x_cholesky_inv_diag), x_cholesky_inv_diag - tf.matmul(tf.transpose(x_cholesky_off_diag), x_smoothed_pair_cov))

        return x_backward_state(x_smoothed_cov, x_smoothed_pair_cov)

    x_smoothed_covs, x_smoothed_pair_covs = tf.scan(update_backward_fn, elems=(x_cholesky_inv_diags, x_cholesky_off_diags), initializer=x_backward_init_state, reverse=True)

    return x_smoothed_covs, x_smoothed_pair_covs


@tf.function(experimental_compile=True)
def covariance_update(x_cholesky_diags, x_cholesky_off_diags):
    return _covariance_update(x_cholesky_diags, x_cholesky_off_diags)


@tf.function
def  _assign_log_lambda(model, batch_kappas, batch_masks, batch_ys, n_monte):

    n_site = model.n_params['n_site']

    alpha_2 = model.alpha_2

    R = tf.shape(batch_ys)[1]
    n_x = model.n_params['n_x']

    n_jackknife = model.n_params['n_jackknife']
    comb_masks = [tf.constant(np.eye(n_monte)[np.array(list(combinations(np.arange(n_monte), n_monte-l)))], tf.float32) for l in range(n_jackknife+1)]
    comb_coefs = tf.constant([(- 1.) ** l * (n_monte - l) ** n_jackknife / np.math.factorial(n_jackknife - l) / np.math.factorial(l) for l in range(n_jackknife+1)], tf.float32)

    def _lambda_g_cost_batch(state, elem):

        kappas, masks, y = elem

        epsx = tf.random.normal((R, 1, n_x), dtype=tf.float32)

        y_not_nan, is_not_nan, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde = compute_eta_loss_1(model, kappas, masks, y)
        with tf.device('/cpu:0'):
            _, x = compute_eta_loss_2(model, x_affine_diag, x_affine_tri, recog_x_prec_tilde, recog_x_mean_dot_prec_tilde, epsx)

        lambda_g_loss, _, _ = compute_lambda_g_loss(model, tf.repeat(x, n_site, axis=1), n_monte)

        lambda_g_cost_epoch = tf.ones(n_site) * - np.inf

        for l in range(n_jackknife+1):

            lambda_g_loss_resampled = tf.tensordot(lambda_g_loss, comb_masks[l], [[1], [2]])
            lambda_g_cost_epoch_resampled = tfp.math.reduce_logmeanexp(tf.reduce_logsumexp(alpha_2 ** - 1 * tfp.math.reduce_logmeanexp(alpha_2 * lambda_g_loss_resampled, axis=-1), axis=0), axis=-1)
            lambda_g_cost_epoch = tfp.math.reduce_weighted_logsumexp([lambda_g_cost_epoch, lambda_g_cost_epoch_resampled], [tf.ones_like(lambda_g_cost_epoch), tf.ones_like(lambda_g_cost_epoch_resampled) * comb_coefs[l]], axis=0)

        n_epoch = tf.reduce_sum(masks, axis=(0, 1))

        return lambda_g_cost_epoch, n_epoch

    lambda_g_cost, n = tf.scan(_lambda_g_cost_batch, elems=(batch_kappas, batch_masks, batch_ys), initializer=(tf.zeros(n_site), tf.zeros(n_site)))
    lambda_g_cost, n = tf.reduce_logsumexp(lambda_g_cost, 0), tf.reduce_sum(n, 0)

    model.log_lambda.assign(tf.math.log(n) - lambda_g_cost)
