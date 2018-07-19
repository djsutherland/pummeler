from __future__ import division, print_function
from collections import Counter
import heapq
import sys
import zipfile

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import geocode_data
from .reader import read_chunks, VERSIONS


def sort_by_region(source, out_fmt, voters_only=True, adj_inc=True,
                   version='2006-10', chunksize=10**5, n_to_sample=5000,
                   stats_only=False):
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

    n_total = 0
    wt_total = 0
    not_in_region = Counter()

    if isinstance(source, list):
        files = source
        opener = open
    else:
        z = zipfile.ZipFile(source, 'r')  # close through GC :|
        opener = z.open
        files = [fn for fn in z.namelist() if fn.endswith('.csv')]

    # We're going to use weighted reservoir sampling to keep a few random rows
    # in memory, so we can estimate the median pairwise distance between
    # features later.
    reservoir = []

    columns = None

    for file in files:
        print("File {}".format(file), file=sys.stderr)

        checked_cols = False
        with opener(file, 'r') as f, tqdm() as bar:
            for chunk in read_chunks(f, voters_only=voters_only,
                                     adj_inc=adj_inc, chunksize=chunksize,
                                     version=version):
                if not checked_cols:
                    if columns is None:
                        columns = list(chunk.columns)

                        cols = set(chunk.columns)
                        extra = cols - all_cols - {'ADJINC', 'ADJINC_orig'}
                        if extra:
                            msg = ("Saw unknown columns; did you pass the "
                                   "right PUMS file version?\n{}")
                            raise ValueError(msg.format(', '.join(extra)))
                        missing = all_cols - cols
                        if missing:
                            msg = ("Didn't see expected columns; did you pass "
                                   "the right PUMS file version?\n{}")
                            raise ValueError(msg.format(', '.join(missing)))
                    else:
                        assert list(chunk.columns) == columns
                    checked_cols = True

                n_total += chunk.shape[0]
                wt_total += chunk.PWGTP.sum()

                # components of mean / std for real-valued features
                reals = chunk[real_feats]
                real_info.append((
                    reals.shape[0] - reals.isnull().sum(),
                    reals.mean(),
                    (reals.astype(np.float128) ** 2).mean()))

                # add onto value counts for discrete features
                for k in discrete:
                    value_counts[k] = chunk[k].value_counts().add(
                        value_counts.get(k, 0), fill_value=0)

                # manage reservoir sample
                rs = np.asarray(
                    np.random.uniform(size=chunk.shape[0]) ** (1 / chunk.PWGTP))
                for r_tup in zip(rs, chunk.itertuples(index=False)):
                    # TODO: could speed this up if it's slow, probably
# maybe there's a weighted version of
# http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
                    if len(reservoir) < n_to_sample:
                        heapq.heappush(reservoir, r_tup)
                    else:
                        heapq.heappushpop(reservoir, r_tup)

                # output into files by region
                if not stats_only:
                    regions = np.empty(chunk.shape[0], dtype=object)
                    for i, tup in enumerate(zip(chunk.ST, chunk.PUMA)):
                        regions[i] = r = puma_to_region.get(tup)
                        if r is None:
                            not_in_region[tup] += 1

                    for r, r_chunk in chunk.groupby(regions):
                        out = out_fmt.format(r)
                        try:
                            mode = 'a' if r in created_files else 'w'
                            r_chunk.to_hdf(
                                out, 'df', format='table', append=True,
                                mode=mode, complib='blosc', complevel=6)
                            old_wt = None
                        except ValueError:
                            # if new chunk has longer strings than previous
                            # one did, this will cause an error...instead of
                            # hardcoding sizes, though, just re-write the data.
                            # (Note that this happens in data validation, i.e.
                            # nothing is written to the file yet.)
                            old = pd.read_hdf(out, 'df')
                            with h5py.File(out, 'r') as f:
                                old_wt = f['total_wt'][()]
                            new = pd.concat([old, r_chunk])
                            new.to_hdf(
                                out, 'df', format='table', mode='w',
                                complib='blosc', complevel=6)

                        with h5py.File(out, 'a') as f:
                            ds = f.require_dataset('total_wt', (), np.int64)
                            if old_wt is None:
                                old_wt = ds[()]
                            ds[()] = old_wt + r_chunk.PWGTP.sum()

                        created_files.add(r)
                bar.update(chunk.shape[0])

    if not_in_region:
        print("Records not in a region:")
        for k, count in not_in_region.most_common():
            print(k, count)

    total = sum(n_nonnan for n_nonnan, mean, mean_sq in real_info)
    real_means = 0
    real_mean_sqs = 0
    for n_nonnan, mean, mean_sq in real_info:
        real_means += n_nonnan / total * mean
        real_mean_sqs += n_nonnan / total * mean_sq

    real_stds = (real_mean_sqs - real_means.astype(np.float128)**2
        ).apply(np.sqrt).astype(np.float64)

    stats = {}
    stats['real_means'] = real_means
    stats['real_stds'] = real_stds
    stats['value_counts'] = value_counts
    stats['n_total'] = n_total
    stats['wt_total'] = wt_total
    stats['version'] = version
    stats['sample'] = pd.DataFrame([t for r, t in reservoir], columns=columns)
    return stats
