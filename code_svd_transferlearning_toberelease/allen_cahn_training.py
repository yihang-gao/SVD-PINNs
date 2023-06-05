import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tools.load_data import load_data_allen_cahn, sol_allen_cahn_eq, rhs_allen_cahn_eq
import pickle

tf.random.set_seed(666)
np.random.seed(666)

import os

os.environ['PYTHONHASHSEED'] = "0"
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def relu3_activation(x):
    x = K.maximum(x ** 3 / 6, 0.0)
    return x


def new_sol(weights_and_bias, U, V, TX):
    layer1 = tf.linalg.matmul(TX, weights_and_bias[1]) + weights_and_bias[2]
    layer1 = tf.maximum(layer1 ** 3 / 6, 0.0)
    # layer1 = tf.math.tanh(layer1)
    layer2 = tf.linalg.matmul(layer1, U)
    layer2 = layer2 * weights_and_bias[0]
    layer2 = tf.linalg.matmul(layer2, V) + weights_and_bias[3]
    layer2 = tf.maximum(layer2 ** 3 / 6, 0.0)
    # layer2 = tf.math.tanh(layer2)
    layer3 = tf.linalg.matmul(layer2, weights_and_bias[4]) + weights_and_bias[5]
    return layer3


def get_new_loss_in(new_nn, d, T_in, X_in_list, Y_in, weights_and_bias, U, V):
    u_xx = 0.0
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as pde_tape1:
        pde_tape1.watch(X_in_list)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as pde_tape2:
            pde_tape2.watch([*X_in_list, T_in])
            u = new_nn(weights_and_bias, U, V, tf.concat([T_in, *X_in_list], axis=1))
        u_x = pde_tape2.gradient(u, X_in_list)
        u_t = pde_tape2.gradient(u, T_in)
    for i in range(d):
        u_xx = u_xx + pde_tape1.gradient(u_x[i], X_in_list[i])
    res = u_t - u_xx - u + u ** 3 - Y_in
    res = res ** 2
    return tf.math.reduce_mean(res)


def get_new_loss_bd(new_nn, TX_bd, Y_bd, weights_and_bias, U, V):
    pred = new_nn(weights_and_bias, U, V, TX_bd)
    err = pred - Y_bd
    err = err ** 2
    return tf.math.reduce_mean(err)


@tf.function
def train_new_step(new_nn, new_optimizer, new_optimizer2, d, balance, TX_bd, Y_bd, T_in, X_in_list, Y_in,
                   weights_and_bias, U, V):
    with tf.GradientTape() as grad_tape:
        grad_tape.watch(weights_and_bias)
        loss = get_new_loss_bd(new_nn, TX_bd, Y_bd, weights_and_bias, U, V) * balance + get_new_loss_in(new_nn, d,
                                                                                                        T_in, X_in_list,
                                                                                                        Y_in,
                                                                                                        weights_and_bias,
                                                                                                        U, V)
        # loss = get_loss_bd(nn, TX_bd, Y_bd)
    grad = grad_tape.gradient(loss, weights_and_bias)
    new_optimizer.apply_gradients(zip(grad[1:], weights_and_bias[1:]))
    new_optimizer2.apply_gradients(zip([grad[0]], [weights_and_bias[0]]))

@tf.function
def train_new_step_sing(new_nn, new_optimizer, new_optimizer2, d, balance, TX_bd, Y_bd, T_in, X_in_list, Y_in,
                   weights_and_bias, U, V):
    with tf.GradientTape() as grad_tape:
        grad_tape.watch(weights_and_bias)
        loss = get_new_loss_bd(new_nn, TX_bd, Y_bd, weights_and_bias, U, V) * balance + get_new_loss_in(new_nn, d,
                                                                                                        T_in, X_in_list,
                                                                                                        Y_in,
                                                                                                        weights_and_bias,
                                                                                                        U, V)
        # loss = get_loss_bd(nn, TX_bd, Y_bd)
    grad = grad_tape.gradient(loss, weights_and_bias)
    # new_optimizer.apply_gradients(zip(grad[1:], weights_and_bias[1:]))
    new_optimizer2.apply_gradients(zip([grad[0]], [weights_and_bias[0]]))


def get_new_error(new_nn, TX_in_test, Y_in_test, weights_and_bias, U, V):
    pred = new_nn(weights_and_bias, U, V, TX_in_test)
    err = pred - Y_in_test
    err = err ** 2
    return tf.math.reduce_mean(err) / tf.math.reduce_mean(Y_in_test ** 2)


### main loop
m = 300
d = 10
c = 50.0
max_new_itr = 20000
check = 100
lr_new_rate = 1e-3
lr_new_rate_D = 1e-2

# you can/may tune and change the  balancing parameter for diffirent pde
balance = 1e2

# we strongly recommend to include weight decaying strategy for all optimizers (both new optimizer and optimizer2) to stablize the training
# new optimizer is for updating all weight matrices except W1
new_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_new_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                         amsgrad=False)

# new optimizer2 is for updating singular values of the weight matrix W1, the simple gradient descent is recommended
# new_optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr_new_rate_D, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
new_optimizer2 = tf.keras.optimizers.SGD(learning_rate=lr_new_rate_D)
# new_optimizer2 = tf.keras.optimizers.RMSprop(learning_rate=lr_new_rate_D, rho=0.5)

# load the pretrained model
with open('pretraining_data/allen_cahn_{}d/weights_and_bias'.format(d), "rb") as fp:
    load_weights_and_bias = pickle.load(fp)

D = tf.Variable(initial_value=load_weights_and_bias[2].numpy(), trainable=True)
U = load_weights_and_bias[3]
V = load_weights_and_bias[4]
w1 = load_weights_and_bias[0]
b1 = load_weights_and_bias[1]
b2 = load_weights_and_bias[5]
w3 = load_weights_and_bias[6]
b3 = load_weights_and_bias[7]
weights_and_bias = [D, w1, b1, b2, w3, b3]

# load the spatial and temperal variable data, here you can also generate new samples
T_in = np.load('pretraining_data/allen_cahn_{}d/T_in.npy'.format(d))
X_in = np.load('pretraining_data/allen_cahn_{}d/X_in_list.npy'.format(d), allow_pickle=True)
# Y_in = np.load('pretraining_data/allen_cahn_{}d/Y_in.npy'.format(d))
TX_bd = np.load('pretraining_data/allen_cahn_{}d/TX_bd.npy'.format(d))
# Y_bd = np.load('pretraining_data/allen_cahn_{}d/Y_bd.npy'.format(d))
TX_test = np.load('pretraining_data/allen_cahn_{}d/TX_test.npy'.format(d))
# Y_test = np.load('pretraining_data/allen_cahn_{}d/Y_test.npy'.format(d))

X_in_list = []
for i in range(d):
    X_in_list.append(X_in[i])

# calculate the right-hand side functions of PDE at the given spatial/temperal variable
# we did data processing to spatial/temperial variables, so we should convert it back.
Y_test = sol_allen_cahn_eq(TX_test[:, 0][:, None] + 1 / 2, TX_test[:, 1:], c=c)
Y_in = rhs_allen_cahn_eq(T_in + 1 / 2, np.concatenate(X_in_list, axis=1), d=d, c=c)
Y_bd = sol_allen_cahn_eq(TX_bd[:, 0][:, None] + 1 / 2, TX_bd[:, 1:], c=c)

TX_test = tf.convert_to_tensor(TX_test, dtype=tf.float32)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)

LOSS = np.zeros(shape=max_new_itr // check + 1)
ERR_test = np.zeros(shape=max_new_itr // check + 1)

num = 0

T_in_batch = tf.convert_to_tensor(T_in, dtype=tf.float32)
for i in range(d):
    X_in_list[i] = tf.convert_to_tensor(X_in_list[i], dtype=tf.float32)
X_in_list_batch = X_in_list
Y_in_batch = tf.convert_to_tensor(Y_in, dtype=tf.float32)
TX_bd_batch = tf.convert_to_tensor(TX_bd, dtype=tf.float32)
Y_bd_batch = tf.convert_to_tensor(Y_bd, dtype=tf.float32)

print('--------------Loading Data Finished----------------- \n')

emp_loss = get_new_loss_bd(new_sol, TX_bd=TX_bd_batch, Y_bd=Y_bd_batch, weights_and_bias=weights_and_bias, U=U,
                           V=V) * balance + get_new_loss_in(new_sol,
                                                            d=d,
                                                            T_in=T_in_batch,
                                                            X_in_list=X_in_list_batch,
                                                            Y_in=Y_in_batch,
                                                            weights_and_bias=weights_and_bias, U=U, V=V)

err = get_new_error(new_sol, TX_test, Y_test, weights_and_bias, U, V)

LOSS[num] = emp_loss
ERR_test[num] = err

print("itr 0, emp loss is {:.3e}, test error is {:.3e}.\n".format(emp_loss, err))

print('--------------Begin Training----------------- \n')
for itr in range(max_new_itr):
    train_new_step(new_sol, new_optimizer, new_optimizer2, d, balance, TX_bd_batch, Y_bd_batch, T_in_batch,
                   X_in_list_batch, Y_in_batch,
                   weights_and_bias, U, V)
    
    # all singular values should be non-negative
    value_D = tf.maximum(weights_and_bias[0], 0.0)
    weights_and_bias[0].assign(value_D)

    if (itr + 1) % check == 0:
        num = num + 1
        # value_D = tf.maximum(weights_and_bias[0], 0.0)
        # weights_and_bias[0].assign(value_D)

        emp_loss = get_new_loss_bd(new_sol, TX_bd=TX_bd_batch, Y_bd=Y_bd_batch, weights_and_bias=weights_and_bias, U=U,
                                   V=V) * balance + get_new_loss_in(new_sol,
                                                                    d=d,
                                                                    T_in=T_in_batch,
                                                                    X_in_list=X_in_list_batch,
                                                                    Y_in=Y_in_batch,
                                                                    weights_and_bias=weights_and_bias, U=U, V=V)

        LOSS[num] = emp_loss

        err = get_new_error(new_sol, TX_test, Y_test, weights_and_bias, U, V)

        ERR_test[num] = err

        print("itr {}, emp loss is {:.3e}, test error is {:.3e}.\n".format(itr + 1, emp_loss, err))

 

# you may save the results
# np.save('results/allen_cahn_{}d/emploss{}_{:.2f}.npy'.format(d, m, c), LOSS)
# np.save('results/allen_cahn_{}d/testerror{}_{:.2f}.npy'.format(d, m, c), ERR_test)
# np.save('results/allen_cahn_{}d/optimizers/emploss{}_{:.2f}_SGD_1e-2.npy'.format(d, m, c), LOSS)
# np.save('results/allen_cahn_{}d/optimizers/testerror{}_{:.2f}_SGD_1e-2.npy'.format(d, m, c), ERR_test)
