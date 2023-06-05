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


initializers = tf.keras.initializers.GlorotUniform()
def get_nn(m=1000, inp_d=2):
    model = Sequential()
    model.add(Dense(m, input_shape=(inp_d,), kernel_initializer=initializers, activation=relu3_activation, use_bias=True))
    model.add(Dense(m, input_shape=(inp_d,),  kernel_initializer=initializers, activation=relu3_activation, use_bias=True))
    model.add(Dense(1,  kernel_initializer=initializers, activation=None, use_bias=True))

    return model


def get_loss_in(nn, d, T_in, X_in_list, Y_in, training=True):
    u_xx = 0.0
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as pde_tape1:
        pde_tape1.watch(X_in_list)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as pde_tape2:
            pde_tape2.watch([*X_in_list, T_in])
            u = nn(tf.concat([T_in, *X_in_list], axis=1), training=training)
        u_x = pde_tape2.gradient(u, X_in_list)
        u_t = pde_tape2.gradient(u, T_in)
    for i in range(d):
        u_xx = u_xx + pde_tape1.gradient(u_x[i], X_in_list[i])
    res = u_t - u_xx - u + u ** 3 - Y_in
    res = res ** 2
    return tf.math.reduce_mean(res)


def get_loss_bd(nn, TX_bd, Y_bd, training=True):
    pred = nn(TX_bd, training=training)
    err = pred - Y_bd
    err = err ** 2
    return tf.math.reduce_mean(err)


@tf.function()
def train_step(nn, d, optimizer, balance, TX_bd, Y_bd, T_in, X_in_list, Y_in):
    with tf.GradientTape() as grad_tape:
        loss = get_loss_bd(nn, TX_bd, Y_bd, training=True) * balance + get_loss_in(nn, d, T_in, X_in_list, Y_in,
                                                                                   training=True)
    grad = grad_tape.gradient(loss, nn.trainable_variables)

    optimizer.apply_gradients(zip(grad, nn.trainable_variables))


def get_minibatch(n1, n2, TX_in, Y_in, TX_bd, Y_bd, batchsize_in=100, batchsize_bd=50):
    idx_in = np.random.choice(n1, batchsize_in, replace=False)
    idx_bd = np.random.choice(n2, batchsize_bd, replace=False)

    return TX_in[idx_in, :], Y_in[idx_in, :], TX_bd[idx_bd, :], Y_bd[idx_bd, :]


def get_error(nn, TX_in_test, Y_in_test, training=False):
    pred = nn(TX_in_test, training=training)
    err = pred - Y_in_test
    err = err ** 2
    return tf.math.reduce_mean(err) / tf.math.reduce_mean(Y_in_test ** 2)


### main loop
m = 300
d = 10
c = 0.0
balance = 1e2
max_itr = 20000
check = 100
lr_rate = 1e-3
sol_pde = get_nn(m=m, inp_d=d + 1)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
new_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
n1, n2, n3, T_in, X_in_list, Y_in, TX_bd, Y_bd, TX_test, Y_test = load_data_allen_cahn(d=d, c=c, n1=1000,
                                                                                       n2=1000,
                                                                                       n_test=10000)
                                                                                       
# please save all data if you first run the code
'''
np.save('pretraining_data/allen_cahn_{}d/T_in'.format(d), T_in)
np.save('pretraining_data/allen_cahn_{}d/X_in_list'.format(d), X_in_list)
# np.save('pretraining_data/allen_cahn_{}d/Y_in'.format(d), Y_in)
np.save('pretraining_data/allen_cahn_{}d/TX_bd'.format(d), TX_bd)
# np.save('pretraining_data/allen_cahn_{}d/Y_bd'.format(d), Y_bd)
np.save('pretraining_data/allen_cahn_{}d/TX_test'.format(d), TX_test)
# np.save('pretraining_data/allen_cahn_{}d/Y_test'.format(d), Y_test)
'''

TX_test = tf.convert_to_tensor(TX_test, dtype=tf.float32)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)

LOSS = np.zeros(shape=max_itr // check + 1)
ERR_test = np.zeros(shape=max_itr // check + 1)
num = 0

print('--------------Loading Data Finished----------------- \n')
T_in_batch = tf.convert_to_tensor(T_in, dtype=tf.float32)
for i in range(d):
    X_in_list[i] = tf.convert_to_tensor(X_in_list[i], dtype=tf.float32)
X_in_list_batch = X_in_list

Y_in_batch = tf.convert_to_tensor(Y_in, dtype=tf.float32)
TX_bd_batch = tf.convert_to_tensor(TX_bd, dtype=tf.float32)
Y_bd_batch = tf.convert_to_tensor(Y_bd, dtype=tf.float32)

LOSS[num] = get_loss_bd(sol_pde, TX_bd=TX_bd_batch, Y_bd=Y_bd_batch, training=False) * balance + get_loss_in(sol_pde,
                                                                                                             d=d,
                                                                                                             T_in=T_in_batch,
                                                                                                             X_in_list=X_in_list_batch,
                                                                                                             Y_in=Y_in_batch,
                                                                                                             training=False)

ERR_test[num] = get_error(nn=sol_pde, TX_in_test=TX_test, Y_in_test=Y_test, training=False)

print('--------------Begin Training----------------- \n')

for itr in range(max_itr):
    train_step(nn=sol_pde, d=d, optimizer=optimizer, balance=balance, TX_bd=TX_bd_batch, Y_bd=Y_bd_batch,
               T_in=T_in_batch, X_in_list=X_in_list_batch,
               Y_in=Y_in_batch)

    if (itr + 1) % check == 0:
        num = num + 1
        emp_loss = get_loss_bd(sol_pde, TX_bd=TX_bd_batch, Y_bd=Y_bd_batch, training=False) * balance + get_loss_in(
            sol_pde,
            d=d,
            T_in=T_in_batch,
            X_in_list=X_in_list_batch,
            Y_in=Y_in_batch,
            training=False)

        LOSS[num] = emp_loss

        err = get_error(nn=sol_pde, TX_in_test=TX_test, Y_in_test=Y_test, training=False)

        ERR_test[num] = err

        print("itr {}, emp loss is {:.3e}, test error is {:.3e}.\n".format(itr + 1, emp_loss, err))


np.save('results/allen_cahn_{}d/emploss{}_full.npy'.format(d, m), LOSS)
np.save('results/allen_cahn_{}d/testerror{}_full.npy'.format(d, m), ERR_test)


# please save all pretrained weights if you first run the code

'''
w1 = sol_pde.trainable_variables[0]
b1 = sol_pde.trainable_variables[1]
w2 = sol_pde.trainable_variables[2]
b2 = sol_pde.trainable_variables[3]
w3 = sol_pde.trainable_variables[4]
b3 = sol_pde.trainable_variables[5]

D, U, V = tf.linalg.svd(w2, full_matrices=True, compute_uv=True)
V = tf.transpose(V)

weights_bias = [w1, b1, D, U, V, b2, w3, b3]

# np.save('pretraining_data/allen_cahn_{}d/weights_and_bias.npy'.format(d), weights_bias)


with open('pretraining_data/allen_cahn_{}d/weights_and_bias'.format(d), "wb") as fp:
    pickle.dump(weights_bias, fp)

'''




'''
with open('pretraining_data/allen_cahn_{}d/weights_and_bias'.format(d), "rb") as fp:
    b = pickle.load(fp)
'''
