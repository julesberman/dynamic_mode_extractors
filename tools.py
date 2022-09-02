
import numpy as np
import scipy.io
import seaborn as sns


def set_seaborn(params={}):
    sns.set()
    sns.set_style('white')
    sns.set_palette("tab10")
    # plt.rcParams['figure.dpi'] = 300


def exp_f(x, e, a): return a*np.exp(x*e)


def build_exp_series(a_s, e_s, noise=0.0, time=np.arange(0, 3, 0.1)):
    components = []
    X = np.zeros_like(time)
    for i in range(len(e_s)):
        c = exp_f(time, e_s[i], a_s[i])
        c = c.real
        X += c
        components.append(c)

    X *= (1+noise * np.random.randn(len(X)))
    y_i = np.argmax(np.abs(e_s.real))
    Y = components[y_i]

    return X, Y, time, components


def mean_square_error(A, B): return np.square(np.subtract(A, B)).mean()


def read_data(file, path='./data/'):
    data = np.array(scipy.io.loadmat(f'{path}{file}')['DATAFILE'])
    ress = data[:, ::2]
    stims = data[:, 1::2]
    ress = np.swapaxes(ress, 0, 1)
    stims = np.swapaxes(stims, 0, 1)

    print('Data provided from: \nLei Zheng 2006 (The Journal of general physiology, 127(5):495–510) \nZheng 2009 (PLoS One, 4(1):e4307)\n Ask permission before usage/reproduction.\n')

    return ress, stims


def build_hankel(data, rows, cols=None):
    if cols is None:
        cols = len(data) - rows
    X = np.empty((rows, cols))
    for k in range(rows):
        X[k, :] = data[k:cols + k]
    return X
