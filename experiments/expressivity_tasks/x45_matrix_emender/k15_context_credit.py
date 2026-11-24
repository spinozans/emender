"""Window-local outcome rows; writes consume an observation, never a prediction label."""
import numpy as np

ARMS = ('global', 'support', 'online', 'stateless', 'writable', 'reset')


class OutcomeRows:
    def __init__(self):
        self.rows = {}

    def read(self, key):
        total, count = self.rows.get(key, (0., 0.))
        return total / (2 + count), count > 0

    def observe(self, key, local, hist, byte):
        if hist.sum():
            p, q = float(local[byte]), float(hist[byte])
            total, count = self.rows.get(key, (0., 0.))
            self.rows[key] = (.9 * total + (q-p)/(q+p), .9 * count + 1)


def evidence(base, counts):
    hist = np.zeros(256, dtype=np.float64)
    for b, n in counts.items():
        hist[b] = n
    support = hist.sum()
    hist /= max(1, support)
    features = [np.log1p(support), hist.max(), base.max(), base[hist.argmax()], hist @ base]
    return hist, features


def design(features, credit, arm):
    f = np.asarray(features); one = np.ones(f.shape[:-1]+(1,))
    if arm == 'global': return one
    if arm in ('support', 'online'):
        offset = .5*f[...,0]+f[...,1]-f[...,2]-f[...,3]
        if arm == 'online': offset = offset+2*credit
        return np.concatenate((one, offset[...,None]), axis=-1)
    c = np.zeros_like(credit) if arm in ('stateless','reset') else credit
    return np.concatenate((one,f,c[...,None]),axis=-1)


def mass(f, credit, arm, weights):
    score = design(f, np.asarray(credit), arm) @ weights
    return (np.asarray(f)[...,0]>0)/(1+np.exp(-score))
