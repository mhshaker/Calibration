import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin

class TempCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, initial_value=1):
        self.initial_value = initial_value
        self.temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)


    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):

        def compute_loss():
            y_pred_model_w_temp = tf.math.divide(X, self.temp)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                        tf.convert_to_tensor(tf.keras.utils.to_categorical(y)), y_pred_model_w_temp))
            return loss

        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        # print(f"Temperature Initial value: {temp.numpy()}")
        for i in range(300):
            opts = optimizer.minimize(compute_loss, var_list=[self.temp])

        return self

    # @property
    # def weights(self):
    #     if self.calibrator_ is not None:
    #         return self.calibrator_.weights_
    #     return self.weights_init

    # @property
    # def coef_(self):
    #     return self.calibrator_.coef_

    # @property
    # def intercept_(self):
    #     return self.calibrator_.intercept_

    def predict_proba(self, prob_x_test):
        return tf.math.divide(prob_x_test, self.temp).numpy()

    def predict(self, prob_x_test):
        calib_prob = self.predict_proba(prob_x_test)
        return calib_prob.argmax(axis=1)
