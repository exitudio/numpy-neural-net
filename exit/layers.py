from abc import ABC, abstractmethod
import numpy as np
from .activations import NoActivation
from .constants import EPSILON
from .initializers import Constant
from .optimizers import Momentum

class Layer(ABC):
    @abstractmethod
    def init_weights(self, num_input, optimizer, initializer):
        pass

    @abstractmethod
    def feed_forward(self, input):
        pass

    @abstractmethod
    def back_prop(self, last_derivative, learning_rate):
        pass


class BatchNorm(Layer):
    """
    I don't know why batch norm does worse than normal layer
    Batch norm backprop
    https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    """
    def init_weights(self, num_output, optimizer, initializer):
        self._batch_norm_G = initializer(1, num_output)
        self._batch_norm_B = initializer(1, num_output)
        # init optimizer
        self._optimizer_G = optimizer.generate_optimizer((1, num_output))
        self._optimizer_B = optimizer.generate_optimizer((1, num_output))
        self._optimizer_mean = Momentum().generate_optimizer((1, num_output)) #optimizer.generate_optimizer((1, num_output))

    def feed_forward(self, z):
        mean = np.mean(z, axis=0)
        mean = self._optimizer_mean.get_velocity(mean)

        self._diff_mean = z-mean
        variance = np.mean(np.square(self._diff_mean), axis=0)

        self._one_over_stddev = 1/np.sqrt(variance + EPSILON)
        self._z_norm = self._diff_mean * self._one_over_stddev
        self._output = self._batch_norm_G * self._z_norm + self._batch_norm_B
        return self._output

    def back_prop(self, last_derivative, learning_rate):
        dG = np.sum(self._z_norm * last_derivative, axis=0)
        dB = np.sum(last_derivative, axis=0)
        self._batch_norm_G -= learning_rate * self._optimizer_G.get_velocity(dG)
        self._batch_norm_B -= learning_rate * self._optimizer_G.get_velocity(dB)

        dz_norm = self._batch_norm_G * last_derivative

        # ---- d_z_minus_u_1 ----
        d_z_minus_u_1 = self._one_over_stddev * dz_norm
        # -----------------------
        d_stddev = -np.square(self._one_over_stddev) * \
            np.sum(self._diff_mean * dz_norm, axis=0)
        d_variance = 0.5 * self._one_over_stddev * d_stddev
        d_z_minus_u_square = np.full(
            self._output.shape, 1/self._output.shape[0]) * d_variance

        # ---- d_z_minus_u_2 ----
        d_z_minus_u_2 = 2 * self._diff_mean * d_z_minus_u_square
        # -----------------------
        d_z_minus_u = d_z_minus_u_1 + d_z_minus_u_2
        dz_1 = 1 * d_z_minus_u
        du = -np.sum(d_z_minus_u, axis=0)

        dz_2 = np.full(self._output.shape, 1/self._output.shape[0]) * du

        dz = dz_1 + dz_2
        return dz


class Dense(Layer):
    def __init__(self, num_output, Activation_function=None, is_batch_norm=False):
        self._num_output = num_output
        self._activation_function = Activation_function(
        ) if Activation_function is not None else NoActivation()
        self._is_batch_norm = is_batch_norm

    def init_weights(self, num_input, optimizer, initializer):
        # init weights
        self._weights = initializer(num_input, self._num_output)
        self._bias = initializer(1, self._num_output)
        # init optimizer
        self._optimizer_w = optimizer.generate_optimizer(self._weights.shape)
        self._optimizer_b = optimizer.generate_optimizer(self._bias.shape)
        # batch_norm
        if self._is_batch_norm:
            self.batch_norm = BatchNorm()
            self.batch_norm.init_weights(self._num_output, optimizer, initializer)

    def feed_forward(self, input):
        z = np.dot(input, self._weights) + self._bias

        if self._is_batch_norm:
            z = self.batch_norm.feed_forward(z)

        # output & save
        output = self._activation_function.feed_forward(z)
        self._input = input
        self._output = output
        return output

    def back_prop(self, last_derivative, learning_rate):
        dz = last_derivative * self._activation_function.back_prop()

        if self._is_batch_norm:
            dz = self.batch_norm.back_prop(dz, learning_rate)

        # Update weight
        # Be careful, it's not mean, it's sum
        dw = np.dot(self._input.T, dz)
        db = np.sum(dz, axis=0)  # np.mean(dz)
        # ------------------------------------------------------

        current_derivative = np.dot(dz, self._weights.T)
        self._weights -= learning_rate * self._optimizer_w.get_velocity(dw)
        self._bias -= learning_rate * self._optimizer_b.get_velocity(db)

        return current_derivative

    @property
    def num_output(self):
        return self._num_output
