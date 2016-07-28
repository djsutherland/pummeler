from __future__ import division, print_function
import sys

import numpy as np
import pandas as pd
import progressbar as pb
from sklearn.metrics.pairwise import euclidean_distances
import six

from .reader import VERSIONS, read_chunks


def get_dummies(df, stats):
    '''
    Gets features for the person records in `df`: standardizes the real-valued
    features, and does one-hot encoding for the discrete ones.
    '''
    info = VERSIONS[stats['version']]

    parts = [((df[info['real_feats']] - stats['real_means'])
              / stats['real_stds']).fillna(0)]
    for k in info['discrete_feats'] + info['alloc_flags']:
        vc = stats['value_counts'][k]
        c = pd.Categorical(df[k], categories=vc.index).codes
        n_codes = len(vc)
        names = ['{}_{}'.format(k, v) for v in vc.index]
        if vc.sum() < stats['n_total']:
            c = c.copy()
            c[c == -1] = n_codes
            n_codes += 1
            s = '{}_nan'.format(k)
            assert s not in names
            names.append(s)
        bit = np.eye(n_codes).take(c, axis=0)
        parts.append(pd.DataFrame(bit, index=df.index, columns=names))
    return pd.concat(parts, axis=1)


def _num_feats(stats):
    n = stats['real_means'].size
    n_total = stats['n_total']
    for k, v in six.iteritems(stats['value_counts']):
        n += v.size + (1 if v.sum() < n_total else 0)
    return n


################################################################################
### Embeddings

def linear_embedding(feats, wts, stats, out=None):
    '''
    Gets the linear kernel embedding (which is just the weighted mean) for
    dummy features `feats`, with sample weighting `wts`.
    '''
    if out is None:
        out = np.empty(feats.shape[1])
    np.dot(wts, feats, out=out)
    out /= wts.sum()
    return out


def rff_embedding(feats, wts, stats, freqs, out=None):
    '''
    Gets the random Fourier feature embedding for dummy features `feats`,
    with sample weighting `wts`.
    '''
    D = freqs.shape[1]
    if out is None:
        out = np.empty(2 * D)

    angles = np.dot(feats, freqs)
    sin_angles = np.sin(angles)  # TODO: could use MKL sincos for this
    cos_angles = np.cos(angles, out=angles)

    np.dot(wts, sin_angles, out=out[:D])
    np.dot(wts, cos_angles, out=out[D:])
    out /= wts.sum()
    return out


def pick_rff_freqs(stats, n_freqs, bandwidth):
    '''
    Sets up sampling with random Fourier features corresponding to a Gaussian
    kernel with the given bandwidth, with an embedding dimension of `2*n_freqs`.
    '''
    n_feats = _num_feats(stats)
    return np.random.normal(0, 1 / bandwidth, size=(n_feats, n_freqs))


def pick_gaussian_bandwidth(stats):
    '''
    Finds the median distance between features from the random sample saved
    in stats.
    '''
    samp = get_dummies(stats['sample'], stats)
    D2 = euclidean_distances(samp, squared=True)
    return np.sqrt(np.median(D2[np.triu_indices_from(D2, k=1)]))


################################################################################

def get_embeddings(files, stats, n_freqs=2048, freqs=None, bandwidth=None):
    n_feats = _num_feats(stats)
    feat_names = None

    if freqs is None:
        if bandwidth is None:
            print("Picking bandwidth by median heuristic...",
                  file=sys.stderr, end='')
            bandwidth = pick_gaussian_bandwidth(stats)
            print("picked {}".format(bandwidth), file=sys.stderr)
        freqs = pick_rff_freqs(stats, n_freqs, bandwidth)
    else:
        n_freqs = freqs.shape[1]

    emb_lin = np.empty((len(files), n_feats))
    emb_rff = np.empty((len(files), 2 * n_freqs))

    bar = pb.ProgressBar(max_value=stats['n_total'])
    read = 0
    bar.start()
    for i, file in enumerate(files):
        df = pd.read_hdf(file)
        feats = get_dummies(df, stats)
        if feat_names is None:
            feat_names = list(feats.columns)
        linear_embedding(feats, df.PWGTP, stats, out=emb_lin[i])
        rff_embedding(feats, df.PWGTP, stats, freqs, out=emb_rff[i])
        read += df.shape[0]
        bar.update(read)
    bar.finish()

    return emb_lin, emb_rff, freqs, bandwidth, feat_names
