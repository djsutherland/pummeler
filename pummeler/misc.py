from itertools import groupby
import os

import h5py
import numpy as np


def get_state_embeddings(embeddings_fn):
    with np.load(embeddings_fn) as d:
        dirname = os.path.dirname(embeddings_fn)
        fm, state_names = build_file_map(d['region_names'], dirname)
        w = get_group_weights(fm)
        ret = {k: w.dot(d[k]) for k in d if k.startswith('emb_')}
        ret['state_names'] = state_names
        return ret


def get_group_weights(file_map):
    '''
    Gives the weights needed to turn a region-level embedding into some
    higher-level embedding (e.g. states).

    file_map should be a list like
        [[(0, 'out/feats_AK_00_01.h5')],
         [(1, 'out/feats_AL_00_01.h5'), (2, 'out/feats_AL_00_02.h5'), ...],
         ...]
    where the integers refer to the order that file appears in the embedding
    matrix.

    Will return a (len(file_map), len(embedding)) array to pre-multiply the
    embeddings by.
    '''
    shape = (len(file_map), max(i for group in file_map for i, k in group) + 1)
    transform = np.zeros(shape)
    for group_i, group in enumerate(file_map):
        for source_i, fn in group:
            with h5py.File(fn, 'r') as f:
                wt = f['total_wt'][()]
            transform[group_i, source_i] = wt
    transform /= transform.sum(axis=1)[:, np.newaxis]
    return transform


def build_file_map(region_names, outdir):
    file_map = []
    state_names = []

    fns = [os.path.join(outdir, 'feats_{}.h5'.format(r)) for r in region_names]
    get_state = lambda pair: region_names[pair[0]].split('_', 1)[0]
    for state, pairs in groupby(enumerate(fns), get_state):
        file_map.append(list(pairs))
        state_names.append(state)
    return file_map, state_names
