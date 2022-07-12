import os
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist
from sklearn.calibration import CalibratedClassifierCV
from betacal import BetaCalibration

dataset = "mnist"
model_path = f"Models/NN_{dataset}"

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

if not os.path.exists(model_path):
  os.makedirs(model_path)


  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(x_train, y_train, epochs=5)
  model.save(model_path)
else:
  print("loading model")
  model = tf.keras.models.load_model(model_path)

# print(model.evaluate(x_test, y_test))

# predict probability
predictions = model.predict(x_test)
print(predictions.shape)

# calibrate
xp = np.argmax(predictions, axis=1)

bc = BetaCalibration(parameters="abm")
bc.fit(xp.reshape(-1, 1), y_test)
