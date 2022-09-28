#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import time


def initial_function(x1, x2, initial_function_selection):
    if initial_function_selection == 1:
        return np.sin(x1 + 0.5 * x2) \
               + 0.5 * x2 * np.cos(2 * x2 ** 2) \
               + 0.01 * x1 ** 4 \
               + 0.01 * (x1 * x2) ** 3
    elif initial_function_selection == 2:
        return -2 * np.sin(x1 / 3 + x2 / 6) \
               + 0.5 * x2 * np.cos(x2 ** 2) \
               - 0.01 * x2 ** 4 \
               - 0.005 * (-(x1 - 0.5) * (x2 - 0.5)) ** 3
    elif initial_function_selection == 3:
        return np.sin(0.5 * x1 + x2) \
               + 0.5 * np.cos(0.1 * (x2 - 3) ** 2) \
               - 0.0001 * ((x2 - 6) - (0.7 * x1) ** 2) ** 4

rng = np.random.RandomState(0)

n_points = 10000
n_train_points = 10000
n_iterations = 0

n_hidden_layers = 6
n_neurons_per_layer = 80

loss = 'mse'
optimizer = 'adam'

x1_test = 4.5 * rng.rand(n_points) - 2.25
x2_test = 4.5 * rng.rand(n_points) - 2.25


tic = time.perf_counter()

initial_function_selection = 1



tf.random.set_seed(0)

model_in = keras.Input(shape=(2,))
model_d1 = layers.Dense(n_neurons_per_layer, activation='relu')(model_in)
model_d2 = layers.Dense(n_neurons_per_layer, activation='relu')(model_d1)
model_d3 = layers.Dense(n_neurons_per_layer, activation='relu')(model_d2)
model_d4 = layers.Dense(n_neurons_per_layer, activation='relu')(model_d3)
model_d5 = layers.Dense(1)(model_d4)
model = keras.Model(inputs=[model_in], outputs=[model_d5])

model.compile(loss=loss, optimizer=optimizer)

y_initial = [initial_function(x1_test[n], x2_test[n],
                              initial_function_selection)
             for n in range(n_points)]
y_initial = np.array(y_initial)


for i in range(n_iterations + 1):
    x1_train = 4.5 * rng.rand(n_train_points) - 2.25
    x2_train = 4.5 * rng.rand(n_train_points) - 2.25

    x_train_transposed = np.transpose(np.array([x1_train, x2_train]))

    y_calculated = np.empty(n_points)

    if i == 0:

        y_train = [initial_function(x1_train[k], x2_train[k],
                                    initial_function_selection)
                   for k in range(n_train_points)]
        y_train = np.array(y_train)

    y_train_transposed = np.transpose(y_train)

    #with nvtx.annotate("data generation", color="purple"):

    model.fit(x_train_transposed, y_train_transposed, epochs=40,
              verbose=0)

for i in range(n_points):

    a = tf.Variable([[4.5 * rng.rand() - 2.25, 4.5 * rng.rand() -
                          2.25]])    
    b = model(a)
    

toc = time.perf_counter()


print(f"Runtime in seconds: {toc - tic:0.3f}")
