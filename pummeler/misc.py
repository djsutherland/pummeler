from itertools import groupby
import os

import h5py
import numpy as np


def get_state_embeddings(embeddings_fn):
    with np.load(embeddings_fn) as d:
        dirname = os.path.dirname(embeddings_fn)

        region_names = d['region_names']
        region_weights = d['region_weights']
        n_subsets = region_weights.shape[1]
        state_names = sorted({r[:2] for r in region_names})
        state_lookup = {n: i for i, n in enumerate(state_names)}

        transform = np.zeros(
            (len(state_names), len(region_names), n_subsets))
        for r_i, (r, w) in enumerate(zip(region_names, region_weights)):
            transform[state_lookup[r[:2]], r_i, :] = w

        state_weights = transform.sum(axis=1)
        transform /= state_weights[:, np.newaxis, :]

        ret = {'state_names': state_names, 'state_weights': state_weights}
        for k in ['bandwidth', 'freqs', 'feature_names', 'subset_queries']:
            if k in d:
                ret[k] = d[k]
        for k in d:
            if k.startswith('emb_'):
                ret[k] = np.einsum('grs, rfs -> gfs', transform, d[k])
        return ret
