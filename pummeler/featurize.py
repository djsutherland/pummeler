from __future__ import division, print_function
import sys

import numpy as np
import pandas as pd
import progressbar as pb
from sklearn.metrics.pairwise import euclidean_distances
import six

from .reader import VERSIONS


def get_dummies(df, stats, num_feats=None, ret_df=True, skip_feats=None,
                dtype=np.float64, out=None):
    '''
    Gets features for the person records in `df`: standardizes the real-valued
    features, and does one-hot encoding for the discrete ones. Skip any
    features in skip_feats.
    '''
    info = VERSIONS[stats['version']]
    skip_feats = set() if skip_feats is None else set(skip_feats)
    if num_feats is None:
        num_feats = _num_feats(stats, skip_feats=skip_feats)

    if out is None:
        out = np.empty((df.shape[0], num_feats), dtype=dtype)
    else:
        assert out.shape == (df.shape[0], num_feats)

    real_feats = [f for f in info['real_feats'] if f not in skip_feats]

    reals = out[:, :len(real_feats)]
    reals[:] = df[real_feats]
    reals[:] -= stats['real_means']
    reals[:] /= stats['real_stds']
    reals[np.isnan(reals)] = 0
    if ret_df:
        feat_names = list(real_feats)
    start_col = len(real_feats)

    for k in info['discrete_feats'] + info['alloc_flags']:
        if k in skip_feats:
            continue
        vc = stats['value_counts'][k]
        c = pd.Categorical(df[k], categories=vc.index).codes
        n_codes = len(vc)
        if ret_df:
            feat_names += ['{}_{}'.format(k, v) for v in vc.index]
        if vc.sum() < stats['n_total']:
            c = c.copy()
            c[c == -1] = n_codes
            n_codes += 1
            if ret_df:
                feat_names.append('{}_nan'.format(k))
        bit = out[:, start_col:start_col + n_codes]
        np.eye(n_codes).take(c, axis=0, out=bit)
        start_col += n_codes
    assert start_col == num_feats

    if ret_df:
        return pd.DataFrame(out, index=df.index, columns=feat_names)
    else:
        return out


def _num_feats(stats, skip_feats=None):
    skip_feats = set() if skip_feats is None else set(skip_feats)
    n_total = stats['n_total']

    n = len(set(stats['real_means'].index) - skip_feats)
    for k, v in six.iteritems(stats['value_counts']):
        if k not in skip_feats:
            n += v.size + (1 if v.sum() < n_total else 0)
    return n


################################################################################
### Embeddings

def linear_embedding(feats, wts, out=None):
    '''
    Gets the linear kernel embedding (which is just the weighted mean) for
    dummy features `feats`, with sample weighting `wts`.
    '''
    if out is None:
        out = np.empty(feats.shape[1])
    np.dot(wts, feats, out=out)
    out /= wts.sum()
    return out


def rff_embedding(feats, wts, freqs, out=None):
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


def pick_rff_freqs(n_freqs, bandwidth, n_feats=None, 
                   stats=None, skip_feats=None):
    '''
    Sets up sampling with random Fourier features corresponding to a Gaussian
    kernel with the given bandwidth, with an embedding dimension of `2*n_freqs`.

    Either pass n_feats, or pass stats (and maybe skip_feats) to compute it.
    '''
    if n_feats is None:
        n_feats = _num_feats(stats, skip_feats=skip_feats)
    return np.random.normal(0, 1 / bandwidth, size=(n_feats, n_freqs))


def pick_gaussian_bandwidth(stats, skip_feats=None):
    '''
    Finds the median distance between features from the random sample saved
    in stats.
    '''
    samp = get_dummies(
        stats['sample'], stats, ret_df=False, skip_feats=skip_feats)
    D2 = euclidean_distances(samp, squared=True)
    return np.sqrt(np.median(D2[np.triu_indices_from(D2, k=1)]))


################################################################################

def get_embeddings(files, stats, n_freqs=2048, freqs=None, bandwidth=None,
                   chunksize=2**13, skip_rbf=False, skip_feats=None):
    skip_feats = set() if skip_feats is None else set(skip_feats)
    n_feats = _num_feats(stats, skip_feats=skip_feats)
    feat_names = None

    if not skip_rbf:
        if freqs is None:
            if bandwidth is None:
                print("Picking bandwidth by median heuristic...",
                      file=sys.stderr, end='')
                bandwidth = pick_gaussian_bandwidth(
                        stats, skip_feats=skip_feats)
                print("picked {}".format(bandwidth), file=sys.stderr)
            freqs = pick_rff_freqs(n_freqs, bandwidth, n_feats=n_feats)
        else:
            n_freqs = freqs.shape[1]

    emb_lin = np.empty((len(files), n_feats))
    if not skip_rbf:
        emb_rff = np.empty((len(files), 2 * n_freqs))

    bar = pb.ProgressBar(max_value=stats['n_total'])
    bar.start()
    read = 0
    dummies = np.empty((chunksize, n_feats))
    for i, file in enumerate(files):
        lin_emb_pieces = []
        if not skip_rbf:
            rff_emb_pieces = []
        weights = []
        total_weight = 0
        for c in pd.read_hdf(file, chunksize=chunksize):
            feats = dummies[:c.shape[0], :]

            if feat_names is None:
                df = get_dummies(c, stats, num_feats=n_feats,
                                 skip_feats=skip_feats, ret_df=True, out=feats)
                feat_names = list(df.columns)
            else:
                get_dummies(c, stats, num_feats=n_feats,
                            skip_feats=skip_feats, ret_df=False, out=feats)

            lin_emb_pieces.append(linear_embedding(feats, c.PWGTP))
            if not skip_rbf:
                rff_emb_pieces.append(rff_embedding(feats, c.PWGTP, freqs))
            w = c.PWGTP.sum()
            weights.append(w)
            total_weight += w

            read += c.shape[0]
            bar.update(read)

        emb_lin[i] = 0
        for w, l in zip(weights, lin_emb_pieces):
            emb_lin[i] += l * (w / total_weight)

        if not skip_rbf:
            emb_rff[i] = 0
            for w, r in zip(weights, rff_emb_pieces):
                emb_rff[i] += r * (w / total_weight)
    bar.finish()

    if skip_rbf:
        return emb_lin, feat_names
    else:
        return emb_lin, emb_rff, freqs, bandwidth, feat_names
