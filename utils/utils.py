import time
import pickle

def format_time(elapsed):
    return time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:15], time.gmtime(elapsed))

def save_pickle(fname, object):
    fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, 'wb') as f:
        pickle.dump(object, f)

def load_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data