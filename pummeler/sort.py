from __future__ import division, print_function
from collections import Counter, defaultdict
import heapq
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import geocode_data
from .reader import read_chunks, VERSIONS, version_info_with_housing


def get_puma_to_region(region_type, puma_year):
    if region_type == "puma_county":
        df = geocode_data(f"puma_region_{puma_year}").region

        def puma_to_region(st_puma):
            st, puma = st_puma
            return df.loc[int(st), int(puma)]

    elif region_type == "puma":
        stab_to_st = geocode_data("state_to_stab").stab.to_dict()
        for stab, st in list(stab_to_st.items()):
            stab_to_st[f"{stab:02}"] = st

        def puma_to_region(st_puma):
            st, puma = st_puma
            return f"{stab_to_st[st]}_{puma}"

    elif region_type == "state":
        stab_to_st = geocode_data("state_to_stab").stab.to_dict()
        for stab, st in list(stab_to_st.items()):
            stab_to_st[f"{stab:02}"] = st

        def puma_to_region(st_puma):
            st, puma = st_puma
            return stab_to_st[st_puma[0]]

    else:
        raise ValueError(f"Bad region_type {region_type!r}")

    return puma_to_region


_ignore_cols = {"ADJINC", "ADJINC_orig", "ADJHSG", "ADJHSG_orig"}


def sort_by_region(
    source,
    out_fmt,
    voters_only=True,
    adj_inc=True,
    adj_hsg=True,
    housing_source=None,  # func from (state, puma) => filename
    housing_cache_size=8,
    version="2006-10",
    chunksize=10 ** 5,
    n_to_sample=5000,
    stats_only=False,
    region_type="puma_county",
    format="parquet",
    add_extension=False,
):
    if housing_source:
        info = version_info_with_housing(version)
    else:
        info = VERSIONS[version]
    all_cols = set(
        info["meta_cols"]
        + info["weight_cols"]
        + info["real_feats"]
        + info["discrete_feats"]
        + info["alloc_flags"]
    )
    real_feats = info["real_feats"]
    discrete = info["discrete_feats"] + info["alloc_flags"]
    wt_col = info["weight_cols"][0]
    assert wt_col.isalpha()  # PWGTP or WGTP, not PWGT6 or whatever

    puma_to_region = get_puma_to_region(region_type, info["region_year"])

    if format.lower() in {"hdf", "hdf5", "h5"}:
        format = "hdf5"
        if add_extension:
            out_fmt += ".h5"
    elif format.lower() in {"parquet", "pq"}:
        format = "parquet"
        if add_extension:
            out_fmt += ".pq"
    else:
        raise ValueError(f"unknown format {format!r}")

    real_info = []  # (num non-nan, mean series, mean of square series) entries
    value_counts = {}  # column name => sum of col.value_counts()

    n_total = 0
    wt_total = 0
    not_in_region = Counter()

    if isinstance(source, list):
        files = source
        opener = open
        sizes = [os.path.getsize(fn) for fn in files]
    else:
        z = zipfile.ZipFile(source, "r")  # close through GC :|
        opener = z.open
        files = [fn for fn in z.namelist() if fn.endswith(".csv")]
        sizes = [z.getinfo(fn).file_size for fn in files]

    # We're going to use weighted reservoir sampling to keep a few random rows
    # in memory, so we can estimate the median pairwise distance between
    # features later.
    reservoir = []

    columns = None

    # We'll write each chunk into a separate small file for now,
    # then merge at the end. (The good file formats for featurization aren't
    # super append-friendly.)
    file_parts = defaultdict(list)

    with tqdm(
        total=sum(sizes),
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    ) as bar:
        for file, size in zip(files, sizes):
            start_size = bar.n
            bar.set_postfix(file=file)

            checked_cols = False
            with opener(file, "r") as in_f:
                for chunk in read_chunks(
                    in_f,
                    voters_only=voters_only,
                    adj_inc=adj_inc,
                    adj_hsg=adj_hsg,
                    housing_source=housing_source,
                    housing_cache_size=housing_cache_size,
                    chunksize=chunksize,
                    version=version,
                ):
                    if not checked_cols:
                        if columns is None:
                            columns = list(chunk.columns)

                            cols = set(chunk.columns)
                            extra = cols - all_cols - _ignore_cols
                            if extra:
                                msg = (
                                    "Saw unknown columns; did you pass the "
                                    "right PUMS file version?\n{}"
                                )
                                raise ValueError(msg.format(", ".join(extra)))
                            missing = all_cols - cols
                            if missing:
                                msg = (
                                    "Didn't see expected columns; did you pass "
                                    "the right PUMS file version?\n{}"
                                )
                                raise ValueError(msg.format(", ".join(missing)))
                        else:
                            assert list(chunk.columns) == columns
                        checked_cols = True

                    n_total += chunk.shape[0]
                    wt_total += chunk[wt_col].sum()

                    # components of mean / std for real-valued features
                    reals = chunk[real_feats]
                    real_info.append(
                        (
                            reals.shape[0] - reals.isnull().sum(),
                            reals.mean(),
                            (reals.astype(np.float128) ** 2).mean(),
                        )
                    )

                    # add onto value counts for discrete features
                    for k in discrete:
                        value_counts[k] = (
                            chunk[k]
                            .value_counts()
                            .add(value_counts.get(k, 0), fill_value=0)
                        )

                    # manage reservoir sample
                    rs = np.asarray(
                        np.random.uniform(size=chunk.shape[0]) ** (1 / chunk[wt_col])
                    )
                    for r_tup in zip(rs, chunk.itertuples(index=False)):
                        # TODO: could speed this up if it's slow, probably
                        # maybe there's a weighted version of
                        # http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
                        if len(reservoir) < n_to_sample:
                            heapq.heappush(reservoir, r_tup)
                        else:
                            heapq.heappushpop(reservoir, r_tup)

                    # output into parts of files by region
                    if not stats_only:
                        regions = np.empty(chunk.shape[0], dtype=object)
                        for i, tup in enumerate(zip(chunk.ST, chunk.PUMA)):
                            regions[i] = r = puma_to_region(tup)
                            if r is None:
                                not_in_region[tup] += 1

                        for r, r_chunk in chunk.groupby(regions):
                            target_name = out_fmt.format(r)
                            parts = file_parts[target_name]
                            fn = f"{target_name}.part{len(parts)+1}"
                            write_chunk(fn, r_chunk, format=format)
                            parts.append(fn)

                    bar.update(start_size + in_f.tell() - bar.n)

    if not_in_region:
        print("Records not in a region:")
        for k, count in not_in_region.most_common():
            print(k, count)

    if not stats_only:
        print("Merging files...")
        dtypes = {k: pd.CategoricalDtype(v.index) for k, v in value_counts.items()}
        for target_file, part_names in tqdm(file_parts.items()):
            merge_chunks(part_names, target_file, format=format, dtypes=dtypes)

    total = sum(n_nonnan for n_nonnan, mean, mean_sq in real_info)
    real_means = 0
    real_mean_sqs = 0
    for n_nonnan, mean, mean_sq in real_info:
        real_means += n_nonnan / total * mean
        real_mean_sqs += n_nonnan / total * mean_sq

    real_stds = (
        (real_mean_sqs - real_means.astype(np.float128) ** 2)
        .apply(np.sqrt)
        .astype(np.float64)
    )

    stats = {}
    stats["real_means"] = real_means
    stats["real_stds"] = real_stds
    stats["value_counts"] = value_counts
    stats["n_total"] = n_total
    stats["wt_total"] = wt_total
    stats["version"] = version
    stats["version_info"] = info
    stats["sample"] = pd.DataFrame([t for r, t in reservoir], columns=columns)
    stats["region_type"] = region_type
    return stats


def write_chunk(fn, df, format):
    if format == "parquet":
        df.to_parquet(fn, row_group_size=65536)
    elif format == "hdf5":
        df.to_hdf(fn, "df", format="table", mode="w", complib="blosc", complevel=6)
    else:
        raise ValueError(f"unknown format {format!r}")


def merge_chunks(in_files, out_fn, format, dtypes):
    if len(in_files) == 1:
        Path(in_files[0]).rename(out_fn)
        return

    if format == "parquet":
        df = pd.concat([pd.read_parquet(fn).astype(dtypes) for fn in in_files])
        df.to_parquet(out_fn, row_group_size=65536)
    elif format == "hdf5":
        df = pd.concat([pd.read_hdf(fn, "df").astype(dtypes) for fn in in_files])
        df.to_hdf(out_fn, "df", format="table", mode="w", complib="blosc", complevel=6)
    else:
        raise ValueError(f"unknown format {format!r}")

    for fn in in_files:
        Path(fn).unlink()
