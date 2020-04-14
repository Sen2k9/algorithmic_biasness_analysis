from six.moves import cPickle as pickle  # for performance
import numpy as np
import Crypto.Random.Fortuna

from pathlib import Path


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def main():
    root = Path(".")

    n = 225

    np.random.seed(5)
    item_number = np.arange(1, n + 1)

    weight = np.random.randint(1, 20, size=n)
    value = np.random.randint(10, 5*n, size=n)
    threshold = 5 * n  # Maximum weight that the bag of thief can hold
    dic = {"item": item_number, "weight": weight,
           "value": value, "threshold": threshold}

    filename = "data_"+str(n)+".pkl"
    my_path = root / "data" / filename

    save_dict(dic, my_path)


    # g_data2 = load_dict('data/data_{}.pkl'.format(n))
    # print(g_data2["item"])
main()
