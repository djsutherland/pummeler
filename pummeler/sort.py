from __future__ import division, print_function
import os
import sys

import numpy as np
import pandas as pd
import progressbar as pb  # should be progressbar2

from .geocode_data import geocode_data
from .reader import read_chunks, VERSIONS


def sort_by_region(files, out_fmt, voters_only=True, adj_inc=True,
                   version='2006-10', chunksize=10**4):
    info = VERSIONS[version]
    all_cols = set(info['meta_cols'] + info['weight_cols'] + info['real_feats']
                   + info['discrete_feats'] + info['alloc_flags'])
    real_feats = info['real_feats']
    discrete = info['discrete_feats'] + info['alloc_flags']

    key = 'puma_region_{}'.format(info['region_year'])
    puma_to_region = geocode_data(key).region.to_dict()

    created_files = set()

    real_info = []  # (num non-nan, mean series, mean of square series) entries
    value_counts = {}  # column name => sum of col.value_counts()

    for file in files:
        print("File {}".format(file), file=sys.stderr)
        bar = pb.ProgressBar(max_value=pb.UnknownLength)
        read = 0
        bar.update(read)

        checked_cols = False
        for chunk in read_chunks(
                file, voters_only=voters_only, adj_inc=adj_inc,
                chunksize=chunksize):
            if not checked_cols:
                cols = set(chunk.columns)
                assert (cols - all_cols).issubset({'ADJINC', 'ADJINC_orig'})
                assert not (all_cols - cols)
                checked_cols = True
            read += chunk.shape[0]

            regions = np.empty(chunk.shape[0], dtype=object)
            for i, tup in enumerate(zip(chunk.ST, chunk.PUMA)):
                regions[i] = puma_to_region.get(tup, np.nan)

            reals = chunk[real_feats]
            real_info.append((
                reals.shape[0] - reals.isnull().sum(),
                reals.mean(),
                (reals.astype(np.float128) ** 2).mean()))

            for k in discrete:
                value_counts[k] = chunk[k].value_counts().add(
                    value_counts.get(k, 0), fill_value=0)

            for r, r_chunk in chunk.groupby(regions):
                out = out_fmt.format(r)
                made = r in created_files
                r_chunk.to_csv(out, mode='a' if made else 'w', header=not made,
                               index=False)
                created_files.add(r)
            bar.update(read)
        bar.finish()

    total = sum(n_nonnan for n_nonnan, mean, mean_sq in real_info)
    real_means = 0
    real_mean_sqs = 0
    for n_nonnan, mean, mean_sq in real_info:
        real_means += n_nonnan / total * mean
        real_mean_sqs += n_nonnan / total * mean_sq

    real_stds = (real_mean_sqs - real_means.astype(np.float128)**2
        ).apply(np.sqrt).astype(np.float64)

    return real_means, real_stds, value_counts
