from __future__ import division, print_function
from collections import defaultdict
from copy import deepcopy
import ctypes
import itertools
from pathlib import Path
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


_cache_needs_nan = {}


def _needs_nan(c, stats):
    try:
        return _cache_needs_nan[c]
    except KeyError:
        vc_sum = stats["value_counts"][c].sum()
        _cache_needs_nan[c] = b = vc_sum < stats["n_total"]
        return b


def get_dummies(
    df, stats, num_feats=None, ret_df=True, skip_feats=None, dtype=np.float64, out=None
):
    """
    Gets features for the person records in `df`: standardizes the real-valued
    features, and does one-hot encoding for the discrete ones. Skip any
    features in skip_feats.
    """
    info = stats["version_info"]
    skip_feats = set() if skip_feats is None else set(skip_feats)

    if out is None:
        if num_feats is None:
            num_feats = _num_feats(stats, skip_feats=skip_feats)
        out = np.empty((df.shape[0], num_feats), dtype=dtype)
    elif num_feats is not None:
        assert out.shape == (df.shape[0], num_feats)

    real_feats = [f for f in info["real_feats"] if f not in skip_feats]

    reals = out[:, : len(real_feats)]
    reals[:] = df[real_feats]
    reals[:] -= stats["real_means"][real_feats].values
    reals[:] /= stats["real_stds"][real_feats].values
    reals[np.isnan(reals)] = 0
    if ret_df:
        feat_names = list(real_feats)
    start_col = len(real_feats)

    for k, vc in stats["value_counts"].items():
        if k in skip_feats:
            continue

        needs_nan = _needs_nan(k, stats)
        n_codes = len(vc) + int(needs_nan)
        _get_dummies(
            df[k], vc, with_nan=needs_nan, out=out[:, start_col : start_col + n_codes]
        )

        if ret_df:
            feat_names += ["{}_{}".format(k, v) for v in vc.index]
            if needs_nan:
                feat_names.append("{}_nan".format(k))

        start_col += n_codes
    assert start_col == out.shape[1]

    if ret_df:
        return pd.DataFrame(out, index=df.index, columns=feat_names)
    else:
        return out


def _get_dummies(col, vc, with_nan, out=None, dtype=np.float64, ordered=None):
    try:
        c = col.cat.codes
    except AttributeError:
        c = pd.Categorical(col, categories=sorted(vc.index.values)).codes
    else:
        assert list(col.cat.categories) == sorted(vc.index.values)
        if ordered is None:
            ordered = col.dtype.ordered

    n_codes = len(vc)
    if with_nan:
        n_codes += 1

    # TODO: cache this per n_codes size?
    if not ordered:
        encodings = np.eye(n_codes)
    else:
        encodings = np.zeros((n_codes, n_codes))
        encodings[np.tril_indices_from(encodings, 0)] = 1
        if with_nan:
            # TODO: allow choosing nan behavior somewhere
            encodings[-1] = 0
            # encodings[-1, :-1] = 0.5
            # encodings[-1, -1] = 1

    if out is None:
        out = np.empty((col.shape[0], n_codes), dtype=dtype)
    encodings.take(c, axis=0, out=out)
    return out


def _num_feats(stats, skip_feats=None):
    skip_feats = frozenset() if skip_feats is None else frozenset(skip_feats)
    n_total = stats["n_total"]

    n = len(set(stats["real_means"].index) - skip_feats)
    for k, v in stats["value_counts"].items():
        if k not in skip_feats:
            n += v.size + (1 if v.sum() < n_total else 0)
    return n


def _feat_names_ids(stats, skip_feats=None):
    skip_feats = frozenset() if skip_feats is None else frozenset(skip_feats)
    info = stats["version_info"]

    names = [f for f in info["real_feats"] if f not in skip_feats]
    ids = names.copy()
    for k, vc in stats["value_counts"].items():
        if k in skip_feats:
            continue
        for v in sorted(vc.index):
            names.append(f"{k}_{v}")
            ids.append(k)
        if _needs_nan(k, stats):
            names.append(f"{k}_nan")
            ids.append(k)
    return names, ids


def _all_feats(stats, inc_meta=False, inc_weights=False):
    info = stats["version_info"]
    res = set(info["real_feats"]) | set(info["discrete_feats"]) | set(info["alloc_flags"])
    if inc_meta:
        res.update(info['meta_cols'])
    if inc_weights:
        res.update(info['weight_cols'])
    return res


def _keeps(identities):
    """
    Figure out which levels we'd want to drop for the sake of
    killing perfect colinearity in dummy features.
    """
    _, ids = np.unique(identities, return_inverse=True)
    starts = np.r_[0, np.diff(ids).nonzero()[0] + 1]
    ends = np.r_[starts[1:], len(identities)]
    keeps = np.ones(len(identities), dtype=bool)
    keeps[starts[ends - starts > 1]] = False
    return keeps


################################################################################
### Embeddings


def read_file_chunks(fn, format=None, columns=None, chunksize=2 ** 30):
    if format is None:
        format = Path(fn).suffix[1:]

    if format in {"hdf5", "hdf", "h5"}:
        yield from pd.read_hdf(fn, chunksize=chunksize, columns=columns)
    elif format in {"parquet", "pq"}:
        from pyarrow.parquet import ParquetFile

        f = ParquetFile(fn)
        next_chunk = []
        size_so_far = 0
        for i in range(f.num_row_groups):
            next_size = f.metadata.row_group(i).num_rows
            if next_chunk and (size_so_far + next_size > chunksize):
                # big enough to yield; will be > chunksize if necessary
                yield f.read_row_groups(
                    next_chunk, columns=columns, use_pandas_metadata=True
                ).to_pandas()
                next_chunk.clear()
                size_so_far = 0

            next_chunk.append(i)
            size_so_far += next_size

        if next_chunk:
            yield f.read_row_groups(
                next_chunk, columns=columns, use_pandas_metadata=True
            ).to_pandas()
    else:
        raise ValueError(f"unknown format {format!r}")


class Featurizer:
    def __init__(self, stats, **skip_kwargs):
        self.stats = stats
        self.skip_feats = self.get_skip_feats(**skip_kwargs)
        self.n_feats = _num_feats(stats, skip_feats=self.skip_feats)

    def get_skip_feats(
        self, only_feats=None, skip_feats=None, skip_alloc_flags=True, **rest
    ):
        if rest:
            warnings.warn(f"Unused arguments: {list(rest.keys())!r}")

        skip_feats = set([] if skip_feats is None else skip_feats)
        if only_feats is not None:
            only_feats = set(only_feats)
            assert only_feats.isdisjoint(skip_feats)
            skip_feats = _all_feats(self.stats) - only_feats
        else:
            if skip_alloc_flags:
                skip_feats.update(self.stats["version_info"]["alloc_flags"])
        return skip_feats

    out_size = None  # subclasses should set

    def set_feat_name_ids(self, names, ids):
        raise NotImplementedError()

    def __call__(self, feats, wts, out=None):
        raise NotImplementedError()


def get_embeddings(
    files,
    stats,
    featurizers,
    chunksize=2 ** 13,
    subsets=None,
    squeeze_queries=True,
    preprocessor=None,
    dtype=np.float64,
):
    if subsets is None:
        n_subsets = 1
    else:
        subsets = subsets.strip()
        if subsets.endswith(","):  # allow trailing comma
            subsets = subsets[:-1]
        n_subsets = subsets.count(",") + 1
        if n_subsets == 1:
            subsets += ","  # make sure eval returns a matrix

        eval_args = {'engine': 'python'}  # :( https://github.com/pandas-dev/pandas/issues/25369

    if preprocessor is None:
        preprocessor = Preprocessor()  # a processor that only tracks skips
    stats = deepcopy(stats)
    preprocessor.handle_stats(stats)

    assert len(featurizers) >= 1
    feat_classes = featurizers
    if len(feat_classes) > 10:
        feat_classes = tqdm(feat_classes, desc="Initializing featurizers", unit="feat")
    featurizers = [feat_class(stats) for feat_class in feat_classes]

    # index of which featurizers need which set of features...
    always_skip = featurizers[0].skip_feats.intersection(
        *(f.skip_feats for f in featurizers[1:])
    )
    preprocessor.always_skip(always_skip)
    to_load = list({"PWGTP"} | preprocessor.need_to_load)

    big_feat_names, big_feat_ids = _feat_names_ids(stats, skip_feats=always_skip)
    each_include = []
    for f in featurizers:
        if f.skip_feats == always_skip:
            each_include.append(slice(None))
            f.set_feat_name_ids(big_feat_names, big_feat_ids)
        else:
            this_wants = set(_feat_names_ids(stats, skip_feats=f.skip_feats)[0])
            inc = np.array([t in this_wants for t in big_feat_names])
            each_include.append(inc)
            f.set_feat_name_ids(
                [big_feat_names[i] for i, v in enumerate(inc) if v],
                [big_feat_ids[i] for i, v in enumerate(inc) if v],
            )

    embeddings = [
        np.empty((len(files), feat.out_size, n_subsets)) for feat in featurizers
    ]
    region_weights = np.empty((len(files), n_subsets))

    dummies = np.empty((chunksize, len(big_feat_names)), dtype=dtype)

    with tqdm(
        total=stats["n_total"], unit="line", unit_scale=True, dynamic_ncols=True
    ) as bar:
        for file_idx, file in enumerate(files):
            bar.set_postfix(file=Path(file).stem)
            weights = []
            total_weights = 0
            emb_pieces = [[] for f in featurizers]

            # get mean embeddings for each chunk in the file
            for c in read_file_chunks(file, chunksize=chunksize, columns=to_load):
                bar.update(c.shape[0])

                preprocessor(c)

                # index into which lines are in which subsets,
                # possibly working around gross pandas bug(?) for one-line dfs
                if subsets is not None:
                    if c.shape[0] == 1:
                        which = pd.concat([c, c]).eval(subsets, **eval_args).astype(bool)[:, :1]
                    else:
                        which = c.eval(subsets, **eval_args).astype(bool)

                    # remove lines not in any subset
                    keep = which.any(axis=0)
                    if not keep.all():
                        c = c.loc[keep]
                        which = which[:, keep]
                        if not c.shape[0]:  # we subsetted away the entire chunk
                            continue

                # expand discrete variables, standardize reals, etc
                if dummies.shape[0] < c.shape[0]:  # if chunksize not supported
                    # .resize() is fidgety...
                    dummies = np.empty((c.shape[0], dummies.shape[1]), dtype=dtype)
                feats = dummies[: c.shape[0]]
                get_dummies(
                    c,
                    stats,
                    skip_feats=always_skip,
                    ret_df=False,
                    dtype=dtype,
                    out=feats,
                )

                # figure out weights within each subset
                wts = np.tile(c.PWGTP.astype(dtype), (n_subsets, 1))
                if subsets is not None:
                    for i, w in enumerate(which):
                        wts[i, ~w] = 0

                # get each set of feature means
                for f, inc, pieces in zip(featurizers, each_include, emb_pieces):
                    pieces.append(f(feats[:, inc], wts))

                # track total weights for later
                ws = wts.sum(axis=1)
                weights.append(ws)
                total_weights += ws

            # figure out the weights for each chunk
            ratios = []
            for ws in weights:
                ratio = np.array(ws, dtype=float, copy=True)
                nz = total_weights != 0
                ratio[nz] /= total_weights[nz]
                ratios.append(ratio)

            # build each embedding with appropriate weight
            for final_embs, emb_pieces in zip(embeddings, emb_pieces):
                final_embs[file_idx] = 0
                for rs, e in zip(ratios, emb_pieces):
                    final_embs[file_idx] += rs * e

            region_weights[file_idx] = total_weights

    if squeeze_queries and n_subsets == 1:
        embeddings = [np.squeeze(e, 2) for e in embeddings]
        region_weights = np.squeeze(region_weights, 1)

    return embeddings, region_weights, featurizers


################################################################################


def linear_embedding(feats, wts, out=None):
    """
    Gets the linear kernel embedding (which is just the weighted mean) for
    dummy features `feats`, with sample weighting `wts`.
    """
    if out is None:
        out = np.empty((feats.shape[1], wts.shape[0]), dtype=feats.dtype)
    np.dot(feats.T, wts.T, out=out)
    w = wts.sum(axis=1)
    nz = w != 0
    out[:, nz] /= w[np.newaxis, nz]
    return out


class LinearFeaturizer(Featurizer):
    def __init__(self, stats, **kwargs):
        super().__init__(stats, **kwargs)
        self.out_size = self.n_feats

    def __call__(self, feats, wts, out=None):
        return linear_embedding(feats, wts, out=out)

    def set_feat_name_ids(self, names, ids):
        self.feat_names = names
        self.feat_ids = ids
        self.keep_multilevels = _keeps(self.feat_ids)


################################################################################

_sincos = None


def _get_sincos(mkl_path=None):
    global _sincos
    if _sincos is None:
        if mkl_path is None:
            mkl_path = os.environ.get("MKL", "libmkl_rt.so")

        try:
            mkl_lib = ctypes.cdll.LoadLibrary(mkl_path)
        except OSError:

            def np_sincos(inp):
                return np.sin(inp), np.cos(inp)

            _sincos = np_sincos
        else:
            in_d = np.ctypeslib.ndpointer(dtype=np.float64)
            out_d = np.ctypeslib.ndpointer(dtype=np.float64, flags="WRITEABLE")
            sincos_d = mkl_lib.vdSinCos
            sincos_d.argtypes = [ctypes.c_int64, in_d, out_d, out_d]
            sincos_d.restype = None

            in_s = np.ctypeslib.ndpointer(dtype=np.float32)
            out_s = np.ctypeslib.ndpointer(dtype=np.float32, flags="WRITEABLE")
            sincos_s = mkl_lib.vsSinCos
            sincos_s.argtypes = [ctypes.c_int64, in_s, out_s, out_s]
            sincos_s.restype = None

            fn = {"<f4": sincos_s, "<f8": sincos_d}

            def mkl_sincos(inp):
                out_sin = np.empty_like(inp)
                out_cos = np.empty_like(inp)
                fn[inp.dtype.str](inp.size, inp, out_sin, out_cos)
                return out_sin, out_cos

            _sincos = mkl_sincos
    return _sincos


def rff_embedding(feats, wts, freqs, out=None):
    """
    Gets the random Fourier feature embedding for dummy features `feats`,
    with sample weighting `wts`.
    """
    D = freqs.shape[1]
    if out is None:
        out = np.empty((2 * D, wts.shape[0]), dtype=feats.dtype)

    sin_angles, cos_angles = _get_sincos()(np.dot(feats, freqs))

    np.dot(sin_angles.T, wts.T, out=out[:D])
    np.dot(cos_angles.T, wts.T, out=out[D:])
    w = wts.sum(axis=1)
    nz = w != 0
    out[:, nz] /= w[np.newaxis, nz] * np.sqrt(D)
    return out


def pick_rff_freqs(
    n_freqs,
    bandwidth,
    seed=None,
    n_feats=None,
    orthogonal=True,
    stats=None,
    skip_feats=None,
):
    """
    Sets up sampling with random Fourier features corresponding to a Gaussian
    kernel with the given bandwidth, with an embedding dimension of `2*n_freqs`.

    Either pass n_feats, or pass stats (and maybe skip_feats) to compute it.

    If orthogonal, uses Orthogonal Random Features:
      https://arxiv.org/abs/1610.09072
    """
    if n_feats is None:
        n_feats = _num_feats(stats, skip_feats=skip_feats)
    rs = np.random.mtrand._rand if seed is None else np.random.RandomState(seed)

    if not orthogonal or n_feats == 1:  # ORF doesn't do anything for d=1
        return rs.normal(0, 1 / bandwidth, size=(n_feats, n_freqs))

    n_reps = int(np.ceil(n_freqs / n_feats))
    freqs = np.empty((n_feats, n_freqs))
    for i in range(n_reps):
        Q, _ = qr(rs.normal(0, 1, size=(n_feats, n_feats)), overwrite_a=True)
        if i < n_reps - 1:
            freqs[:, i * n_feats : (i + 1) * n_feats] = Q.T
        else:
            freqs[:, i * n_feats :] = Q[: n_freqs - i * n_feats].T

    S = rs.chisquare(n_feats, size=n_freqs)
    np.sqrt(S, out=S)
    S /= bandwidth
    freqs *= S[np.newaxis, :]
    return freqs


def pick_gaussian_bandwidth(stats, skip_feats=None):
    """
    Finds the median distance between features from the random sample saved
    in stats.
    """
    return _get_median(
        get_dummies(stats["sample"], stats, ret_df=False, skip_feats=skip_feats)
    )


def _get_median(samp):
    D2 = euclidean_distances(samp, squared=True)
    return np.sqrt(np.median(D2[np.triu_indices_from(D2, k=1)]))


class RFFFeaturizer(Featurizer):
    def __init__(
        self,
        stats,
        n_freqs=2048,
        freqs=None,
        bandwidth=None,
        orthogonal=True,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(stats, **kwargs)
        self.dtype = dtype
        self.out_size = 2 * n_freqs
        self.bandwidth = bandwidth

        if freqs is None:
            if bandwidth is None:
                # print(
                #     "Picking bandwidth by median heuristic...", file=sys.stderr, end=""
                # )
                self.bandwidth = bandwidth = pick_gaussian_bandwidth(
                    stats, skip_feats=self.skip_feats
                )
                # print("picked {}".format(bandwidth), file=sys.stderr)
            freqs = pick_rff_freqs(
                n_freqs,
                bandwidth,
                seed=seed,
                n_feats=_num_feats(stats, skip_feats=self.skip_feats),
                orthogonal=orthogonal,
            )
        if dtype is not None:
            freqs = freqs.astype(dtype)
        self.freqs = freqs

    def set_feat_name_ids(self, names, ids):
        self.input_feat_names = names
        self.input_feat_ids = ids
        self.keep_multilevels = np.ones(self.out_size, dtype=np.bool)

    def __call__(self, feats, wts, out=None):
        if self.dtype is None:
            self.dtype = feats.dtype
            self.freqs = self.freqs.astype(self.dtype)

        return rff_embedding(feats, wts, self.freqs, out=out)


################################################################################


class MyAdditiveExtras(Featurizer):
    # requires my_proc.MyPreprocessor
    def __init__(self, stats, seed=None, **kwargs):
        super().__init__(stats, **kwargs)

        # We're doing:
        #   Mean (linear) features for everything.
        #   Additional RFF features for:
        self.rff_reals = ["AGEP", "PINCP", "WKHP"]
        self.one_n_freqs = 128
        self.pair_n_freqs = 256
        #   Interaction features between all pairs of those and
        self.discretes = ["SEX", "ETHNICITY", "HASDEGREE"]

        for d in self.discretes:
            assert d not in self.skip_feats

        self.rff_pairs = list(itertools.combinations(self.rff_reals, 2))
        self.rff_discrete_pairs = [
            (r, d) for r in self.rff_reals for d in self.discretes
        ]
        self.discrete_pairs = list(itertools.combinations(self.discretes, 2))

        vcs = stats["value_counts"]
        _levels = {}

        def levels(d):
            if d not in _levels:
                _levels[d] = lev = ["{}".format(v) for v in vcs[d].index]
                if _needs_nan(d, stats):
                    lev.append("nan")
            return _levels[d]

        self.feat_names, self.feat_identities = zip(
            *(
                [
                    (f"{r}_{sc}{i}", f"{r}_rff")
                    for r in self.rff_reals
                    for sc in ["sin", "cos"]
                    for i in range(self.one_n_freqs)
                ]
                + [
                    (f"{d1}_{d2}_{sc}{i}", f"{d1}_{d2}_rff")
                    for d1, d2 in self.rff_pairs
                    for sc in ["sin", "cos"]
                    for i in range(self.pair_n_freqs)
                ]
                + [
                    (f"{d}_{r}_{v}_{sc}{i}", f"{d}_{r}_rff")
                    for r, d in self.rff_discrete_pairs
                    for sc in ["sin", "cos"]
                    for i in range(self.one_n_freqs)
                    for v in levels(d)
                ]
                + [
                    (f"{d1}_{d2}_{v1}_{v2}", f"{d1}_{d2}")
                    for d1, d2 in self.discrete_pairs
                    for v1 in levels(d1)
                    for v2 in levels(d2)
                ]
            )
        )
        self.out_size = len(self.feat_names)

        samp = np.array(stats["sample"][self.rff_reals], copy=True)
        samp -= stats["real_means"][self.rff_reals][np.newaxis, :]
        samp /= stats["real_stds"][self.rff_reals][np.newaxis, :]
        samp[np.isnan(samp)] = 0

        self.one_bws = [_get_median(samp[:, [i]]) for i in range(len(self.rff_reals))]
        self.pair_bws = [
            _get_median(samp[:, [i, j]])
            for i, j in itertools.combinations(range(len(self.rff_reals)), 2)
        ]

        rs = np.random.mtrand._rand if seed is None else np.random.RandomState(seed + 1)
        self.one_freqs = [
            pick_rff_freqs(
                self.one_n_freqs, bandwidth=bw, n_feats=1, seed=rs.randint(2 ** 23)
            )
            for bw in self.one_bws
        ]
        self.one_freqs_d = {n: f for n, f in zip(self.rff_reals, self.one_freqs)}
        self.pair_freqs = [
            pick_rff_freqs(
                self.pair_n_freqs, bandwidth=bw, n_feats=2, seed=rs.randint(2 ** 23)
            )
            for bw in self.pair_bws
        ]

        # only want to drop multilevels for discrete-discrete features
        self.keep_multilevels = _keeps(self.feat_identities)
        for i in (~self.keep_multilevels).nonzero()[0]:
            if self.feat_names[i].endswith("_rff"):
                self.keep_multilevels[i] = True

    def set_feat_name_ids(self, names, ids):
        self.feat_locs = f = defaultdict(list)
        for i, s in enumerate(ids):
            f[s].append(i)

    def __call__(self, feats, wts, out=None):
        pos = 0
        n_subs, n = wts.shape

        if out is None:
            out = np.empty((self.out_size, n_subs))

        # real-only features
        step = 2 * self.one_n_freqs
        for f, freqs in zip(self.rff_reals, self.one_freqs):
            rff_embedding(
                feats[:, self.feat_locs[f]], wts, freqs, out=out[pos : pos + step]
            )
            pos += step

        # real pair features
        step = 2 * self.pair_n_freqs
        for (f1, f2), freqs in zip(self.rff_pairs, self.pair_freqs):
            rff_embedding(
                feats[:, self.feat_locs[f1] + self.feat_locs[f2]],
                wts,
                freqs,
                out=out[pos : pos + step],
            )
            pos += step

        # real-discrete pair features
        # want to make weight subsets that only hit the relevant feats
        for fr, fd in self.rff_discrete_pairs:
            d = feats[:, self.feat_locs[fd]]
            l = d.shape[1]
            freqs = self.one_freqs_d[fr]
            step = 2 * self.one_n_freqs * l

            d_wts = d.T[np.newaxis, :, :] * wts[:, np.newaxis, :]  # n_subs, l, n
            d_wts = d_wts.reshape(l * n_subs, n)
            tmp_out = rff_embedding(
                feats[:, self.feat_locs[fr]], d_wts, freqs
            )  # D, l * n_subs
            out[pos : pos + step] = tmp_out.reshape(-1, n_subs)
            pos += step

        # discrete interactions
        for f1, f2 in self.discrete_pairs:
            d1 = feats[:, self.feat_locs[f1]]
            d2 = feats[:, self.feat_locs[f2]]
            l1 = d1.shape[1]
            l2 = d2.shape[1]
            step = l1 * l2
            d = (d1[:, :, np.newaxis] * d2[:, np.newaxis, :]).reshape(n, step)
            linear_embedding(d, wts, out=out[pos : pos + step])
            pos += step

        assert pos == self.out_size
        return out


################################################################################


class Preprocessor:
    def handle_stats(self, stats):
        assert not hasattr(self, "stats")
        self.stats = stats
        stats["version"] += "_processed"
        self.skip = set()

    def always_skip(self, skip_feats):
        self.skip.update(skip_feats)

    @property
    def need_to_load(self):
        return _all_feats(self.stats, inc_meta=True, inc_weights=True) - self.skip

    def __call__(self, df):
        pass
