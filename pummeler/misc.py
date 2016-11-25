from __future__ import print_function, division
import sys

import numpy as np

from .data import geocode_data


def _get_merged_embeddings(data_dict, mapping_fn, out_prefix):
    region_names = data_dict['region_names']
    region_weights = data_dict['region_weights']

    squeezed = region_weights.ndim == 1
    if squeezed:
        region_weights = region_weights[:, np.newaxis]

    n_subsets = region_weights.shape[1]

    mapped_names = [mapping_fn(r) for r in region_names]
    m_names = sorted(set(mapped_names))
    m_names_lookup = {n: i for i, n in enumerate(m_names)}

    transform = np.zeros(
        (len(m_names), len(region_names), n_subsets))
    for r_i, (m, w) in enumerate(zip(mapped_names, region_weights)):
        transform[m_names_lookup[m], r_i, :] = w

    m_weights = transform.sum(axis=1)
    transform /= m_weights[:, np.newaxis, :]

    ret = {'{}_names'.format(out_prefix): m_names,
           '{}_weights'.format(out_prefix): m_weights}
    for k in data_dict:
        if k.startswith('emb_'):
            print("Mapping {}...".format(k), end='', file=sys.stderr)
            emb = data_dict[k]
            if squeezed:
                emb = emb[:, :, np.newaxis]
            ret[k] = np.einsum('grs, rfs -> gfs', transform, emb)
            if squeezed:
                ret[k] = ret[k][:, :, 0]
            print("done", file=sys.stderr)
        elif k in {'region_names', 'region_weights'}:
            pass
        else:
            ret[k] = data_dict[k]
    return ret


def get_state_embeddings(data_dict):
    state_mapper = lambda r: r[:2]
    return _get_merged_embeddings(data_dict, state_mapper, 'state')


def get_merged_embeddings(data_dict):
    fn = geocode_data('region_superregion').merged_region.to_dict().__getitem__
    return _get_merged_embeddings(data_dict, fn, 'merged')
