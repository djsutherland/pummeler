from __future__ import print_function, division

import numpy as np

from .data import geocode_data


def merge_embeddings(embeddings, region_weights, region_maps):
    squeezed = region_weights.ndim == 1
    if squeezed:
        region_weights = region_weights[:, np.newaxis]

    n_subsets = region_weights.shape[1]

    m_names = sorted(set(region_maps))
    m_names_lookup = {n: i for i, n in enumerate(m_names)}

    transform = np.zeros((len(m_names), embeddings[0].shape[0], n_subsets))
    for r_i, (m, w) in enumerate(zip(region_maps, region_weights)):
        transform[m_names_lookup[m], r_i, :] = w

    m_weights = transform.sum(axis=1)

    # normalize transform so that its sum along axis 1 is 1
    # this is kind of gross to allow for zero sums...maybe there's a better way
    nz = np.broadcast_to((m_weights != 0)[:, np.newaxis, :], transform.shape)
    transform[nz] /= np.broadcast_to(m_weights[:, np.newaxis, :], transform.shape)[nz]

    m_embeddings = []
    for emb in embeddings:
        if squeezed:
            emb = emb[:, :, np.newaxis]

        # need to do a matrix multiply for each subset:
        #  - np.einsum('grs,rfs->gfs') would do this, but doesn't call BLAS
        #  - rolling the subset axis to the front and calling np.matmul
        #    would do this, but it just calls einsum anyway:
        #    https://github.com/numpy/numpy/issues/7569

        out = np.empty((n_subsets, len(m_names), emb.shape[1]))
        for i in range(n_subsets):
            np.dot(transform[:, :, i], emb[:, :, i], out=out[i])
        m_emb = np.rollaxis(out, 0, 3)
        if squeezed:
            m_emb = m_emb[:, :, 0]
        m_embeddings.append(m_emb)

    return m_embeddings, m_names, m_weights


def _get_merged_embeddings(data_dict, mapping_fn, out_prefix):
    emb_names, embeddings = zip(
        *[(k, v) for k, v in data_dict.items() if k.startswith("emb_")]
    )

    m_embeddings, m_names, m_weights = merge_embeddings(
        embeddings,
        data_dict["region_weights"],
        [mapping_fn(r) for r in data_dict["region_names"]],
    )

    ret = data_dict.copy()
    ret[f"{out_prefix}_names"] = m_names
    ret[f"{out_prefix}_weights"] = m_weights
    for n, e in zip(emb_names, m_embeddings):
        ret[n] = e
    return ret


def get_state_embeddings(data_dict):
    state_mapper = lambda r: r[:2]
    return _get_merged_embeddings(data_dict, state_mapper, "state")


def get_merged_embeddings(data_dict):
    fn = geocode_data("region_superregion").merged_region.to_dict().__getitem__
    return _get_merged_embeddings(data_dict, fn, "merged")
