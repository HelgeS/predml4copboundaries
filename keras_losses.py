from keras import backend as K


# Adapted from https://github.com/lukovkin/linex-keras/blob/master/linex_loss.py
def linex_underestimate_fn(factor=1):
    return lambda y_true, y_pred: linex_loss_val(y_true, y_pred, factor)


def linex_overestimate_fn(factor=1):
    return lambda y_true, y_pred: linex_loss_val(y_true, y_pred, -factor)


def linex_underestimate(y_true, y_pred):
    return linex_loss_val(y_true, y_pred, 1)


def linex_overestimate(y_true, y_pred):
    return linex_loss_val(y_true, y_pred, -1)


def linex_loss_val(y_true, y_pred, a):
    delta = sign_ae(y_true, y_pred)
    # delta = K.abs(y_true - y_pred)
    res = linex_loss(delta, a=a)
    return K.mean(res)


def linex_loss(delta, a=-1, b=1):
    """
    LinEx(eps) = b * (e^(a * eps) - a * eps - 1)

    A. Zellner, “Bayesian Estimation and Prediction Using Asymmetric Loss Functions,”
    J. Am. Stat. Assoc., vol. 81, no. 394, pp. 446–451, 1986.
    :param delta:
    :param a:
    :param b:
    :return:
    """
    if a != 0 and b > 0:
        return b * (K.exp(a * delta) - a * delta - 1)
    else:
        raise ValueError


def sign_ae(x, y):
    sign_x = K.sign(x)
    sign_y = K.sign(y)
    delta = x - y
    return sign_x * sign_y * K.abs(delta)


def shifted_mse_both(y_true, y_pred):
    overest = shifted_mse_overestimate(y_true[:, 0], y_pred[:, 0])
    underest = shifted_mse_underestimate(y_true[:, 1], y_pred[:, 1])
    return (overest + underest)/2.0


def shifted_mse_underestimate_fn(factor):
    return lambda y_true, y_pred: shifted_mse(y_true, y_pred, factor)


def shifted_mse_overestimate_fn(factor):
    return lambda y_true, y_pred: shifted_mse(y_true, y_pred, -factor)


def shifted_mse_underestimate(y_true, y_pred):
    return shifted_mse(y_true, y_pred, 1)


def shifted_mse_overestimate(y_true, y_pred):
    return shifted_mse(y_true, y_pred, -1)


def shifted_mse(y_true, y_pred, a):
    if a == 0:
        raise ValueError

    x = y_pred - y_true #K.abs(y_true - )
    return K.pow(x, 2) * K.pow(K.sign(x) + a, 2)


def pe_ann_both(y_true, y_pred):
    overest = pe_ann_overestimate(y_true[:, 0], y_pred[:, 0])
    underest = pe_ann_underestimate(y_true[:, 1], y_pred[:, 1])
    return (overest + underest)/2.0


def pe_ann_underestimate_fn(factor):
    return lambda y_true, y_pred: pe_ann(y_true, y_pred, 1, factor)


def pe_ann_overestimate_fn(factor):
    return lambda y_true, y_pred: pe_ann(y_true, y_pred, 1, -factor)


def pe_ann_underestimate(y_true, y_pred):
    return pe_ann(y_true, y_pred, 1, 2)


def pe_ann_overestimate(y_true, y_pred):
    return pe_ann(y_true, y_pred, 1, -2)


def pe_ann(y_true, y_pred, a, b):
    x = y_pred - y_true
    return (a + 1/(1 + K.exp(-b*x))) * x**2
