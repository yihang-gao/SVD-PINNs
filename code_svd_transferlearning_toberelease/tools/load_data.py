import numpy as np
import tensorflow as tf


def sol_allen_cahn_eq(T, X, c=0.0):
    X_md = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    Y = np.sin(np.pi / 2 * np.abs((1 - X_md)) ** (2.5)) * np.exp(-T) + c * np.sin(np.pi / 2 * (1 - X_md)) * np.exp(-T)
    return Y


def rhs_allen_cahn_eq(T, X, d, c=0.0):
    X_md = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    stor = np.abs(1 - X_md)
    Y = np.sin(np.pi / 2 * stor ** (2.5)) * np.exp(-T) + c * np.sin(np.pi / 2 * stor) * np.exp(-T)
    uxx = np.exp(-T) * (-np.sin(np.pi / 2 * stor ** (2.5)) * (np.pi / 2 * 2.5 * (stor ** (1.5))) ** 2 + np.cos(
        np.pi / 2 * stor ** (2.5)) * np.pi / 2 * 2.5 * 1.5 * stor ** (0.5) - np.cos(
        np.pi / 2 * stor ** (2.5)) * np.pi / 2 * stor ** (1.5) * 2.5 * (d - 1) / X_md) - np.exp(-T) * (
                  np.pi / 2) * c * (
                  np.pi / 2 * np.sin(np.pi / 2 * stor) + np.cos(np.pi / 2 * stor) * (d - 1) / X_md)
    Y = Y ** 3 - 2 * Y - uxx

    return Y


def gen_ball(num=100, d=2, radius=1.0):
    X = np.random.normal(size=(num, d))
    X_md = np.sqrt(np.sum(X ** 2, axis=1))
    X_md = X_md[:, None]
    X = X / X_md
    r = np.random.uniform(low=-1 * radius, high=1.0 * radius, size=(num, 1))
    X = X * r
    return X


def gen_sphere(num=100, d=2, radius=1.0):
    X = np.random.normal(size=(num, d))
    X_md = np.sqrt(np.sum(X ** 2, axis=1))
    X_md = X_md[:, None]
    X = X / X_md
    return X


def load_data_allen_cahn(d=2, c=0.0, n1=1000, n2=1000, n_test=10000):
    T_mean = 1 / 2

    X1 = gen_ball(num=n1, d=d, radius=1.0)
    T1 = np.random.uniform(0, 1, size=(n1, 1))
    Y1 = rhs_allen_cahn_eq(T1, X1, d=d, c=c)
    T1 = T1 - T_mean

    n2 = n2 // 2

    X2 = gen_sphere(num=n2, d=d, radius=1.0)
    T2 = np.random.uniform(0, 1, size=(n2, 1)) - T_mean
    TX2 = np.concatenate((T2, X2), axis=1)
    Y2 = np.zeros(shape=(n2, 1))

    X3 = gen_ball(num=n2, d=d, radius=1.0)
    T3 = np.zeros(shape=(n2, 1))
    Y3 = sol_allen_cahn_eq(T3, X3, c=c)
    T3 = T3 - T_mean
    TX3 = np.concatenate((T3, X3), axis=1)

    n2 = n2 * 2

    X0 = gen_ball(num=n_test, d=d, radius=1.0)
    T0 = np.random.uniform(0, 1, size=(n_test, 1))
    Y0 = sol_allen_cahn_eq(T0, X0, c=c)
    T0 = T0 - T_mean
    TX0 = np.concatenate((T0, X0), axis=1)

    TX_bd = np.concatenate((TX2, TX3), axis=0)
    Y_bd = np.concatenate((Y2, Y3), axis=0)
    T_in = T1
    Y_in = Y1

    TX_test = TX0
    Y_test = Y0

    X_in_list = []
    for i in range(d):
        X_in_list.append(X1[:, i][:, None])

    return n1, n2, n_test, T_in, X_in_list, Y_in, TX_bd, Y_bd, TX_test, Y_test


def sol_linear_parabolic_eq(T, X, c=0.0):
    X_md = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    Y = np.exp(X_md * (np.sqrt(np.abs(1 - T)) + c * (1 - T)))
    return Y


def rhs_linear_parabolic_eq(T, X, d, c=0.0):
    X_md = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True)) + 1e-7
    TT = np.sqrt(np.abs(1 - T)) + c * (1 - T)
    Y = (1 + X_md / 2) * np.exp(TT * X_md) * (TT ** 2 + TT * (d - 1) / X_md) + np.exp(TT * X_md) * TT / 2
    u_t = - np.exp(TT * X_md) * X_md * (1 / np.sqrt(np.abs(1 - T) + 1e-9) + c)
    return - Y + u_t


def gen_ball0(num=100, d=2, radius=1.0):
    X = np.random.normal(size=(num, d))
    X_md = np.sqrt(np.sum(X ** 2, axis=1))
    X_md = X_md[:, None]
    X = X / X_md
    r = np.random.uniform(low=1e-3 * radius, high=1.0 * radius, size=(num, 1))
    X = X * r
    return X


def load_data_linear_parabolic(d=2, c=0.0, n1=1000, n2=1000, n_test=10000):
    T_mean = 1 / 2

    X1 = gen_ball0(num=n1, d=d, radius=1.0)
    T1 = np.random.uniform(0, 1 - 0.5, size=(n1, 1))
    Y1 = rhs_linear_parabolic_eq(T1, X1, d=d, c=c)
    T1 = T1 - T_mean

    n2 = n2 // 2

    X2 = gen_sphere(num=n2, d=d, radius=1.0 - 1e-5)
    T2 = np.random.uniform(0, 1 - 0.5, size=(n2, 1))
    Y2 = sol_linear_parabolic_eq(T2, X2, c=c)
    T2 = T2 - T_mean
    TX2 = np.concatenate((T2, X2), axis=1)

    X3 = gen_ball0(num=n2, d=d, radius=1.0)
    T3 = np.zeros(shape=(n2, 1))
    Y3 = sol_linear_parabolic_eq(T3, X3, c=c)
    T3 = T3 - T_mean
    TX3 = np.concatenate((T3, X3), axis=1)

    n2 = n2 * 2

    X0 = gen_ball0(num=n_test, d=d, radius=1.0)
    T0 = np.random.uniform(0, 1 - 0.5, size=(n_test, 1))
    Y0 = sol_linear_parabolic_eq(T0, X0, c=c)
    T0 = T0 - T_mean
    TX0 = np.concatenate((T0, X0), axis=1)

    TX_bd = np.concatenate((TX2, TX3), axis=0)
    Y_bd = np.concatenate((Y2, Y3), axis=0)
    T_in = T1
    Y_in = Y1

    TX_test = TX0
    Y_test = Y0

    X_in_list = []
    for i in range(d):
        X_in_list.append(X1[:, i][:, None])

    return n1, n2, n_test, T_in, X_in_list, Y_in, TX_bd, Y_bd, TX_test, Y_test


def sol_elliptic_eq(X, c=0.0):
    X_md = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    mid = np.abs(1 - X_md)
    Y = np.sin(np.pi / 2 * mid ** (2.5) + c * np.pi / 2 * mid ** 2)
    return Y


def rhs_elliptic_eq(X, d, c=0.0):
    X_md = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True)) + 1e-7
    stor = np.abs(1 - X_md)
    stor1 = np.pi / 2 * stor ** (2.5) + c * np.pi / 2 * stor
    stor2 = np.pi / 2 * (2.5 * stor ** (1.5) + c * 2.0 * stor)
    a_x = 1 + np.sum(X ** 2, axis=1, keepdims=True) / 2
    u_x2 = (np.cos(stor1) * stor2) ** 2
    u_xx = - np.sin(stor1) * stor2 ** 2 + np.cos(stor1) * (
            np.pi / 2 * 2.5 * 1.5 * stor ** (0.5) + np.pi / 2 * c) - np.cos(stor1) * stor2 * (d - 1) / X_md

    return np.cos(stor1) * stor2 * X_md - a_x * u_xx + u_x2


def load_data_elliptic(d=2, c=0.0, n1=1000, n2=1000, n_test=10000):
    X1 = gen_ball(num=n1, d=d, radius=1.0)
    Y1 = rhs_elliptic_eq(X1, d=d, c=c)

    X2 = gen_sphere(num=n2, d=d, radius=1.0)
    Y2 = rhs_elliptic_eq(X2, c=c)

    X0 = gen_ball(num=n_test, d=d, radius=1.0)
    Y0 = rhs_elliptic_eq(X0, c=c)

    X_bd = X2
    Y_bd = Y2
    Y_in = Y1

    X_test = X0
    Y_test = Y0

    X_in_list = []
    for i in range(d):
        X_in_list.append(X1[:, i][:, None])

    return n1, n2, n_test, X_in_list, Y_in, X_bd, Y_bd, X_test, Y_test
