from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import six

from .reader import VERSIONS


_format_info = {
    fmt: (frozenset(formats), exts)
    for fmt, formats, exts in [
        ("pickle", ["pickle"], [".pkl", ".pickle"]),
        ("npz", ["npz"], [".npz"]),
        ("hdf5", ["hdf5", "hdf", "h5"], [".h5", ".hdf5"]),
    ]
}


def _normalize_format(format, ext=None, default=None):
    for fmt, (fmts, exts) in _format_info.items():
        if format in fmts or (format is None and ext in exts):
            return fmt

    if format is None:
        if default is not None:
            return default
        raise ValueError(f"couldn't guess format from extension {ext}; pass format")
    else:
        raise ValueError(f"unknown format {format}")


def _get_fn_format(fn, format=None):
    fn = Path(fn)

    if not fn.exists():
        for fmt, (fmts, exts) in _format_info.items():
            if format is None or format in fmts:
                for ext in exts:
                    if fn.with_suffix(ext).exists():
                        return fn.with_suffix(ext), fmt
        raise OSError(f"file {fn} doesn't exist")

    return fn, _normalize_format(format, fn.suffix)


def save_stats(fn, stats, format=None, add_suffix=True):
    fn = Path(fn)
    format = _normalize_format(format, ext=fn.suffix, default="pickle")
    if not fn.suffix and add_suffix:
        fn = fn.with_suffix(next(iter(_format_info[format][1])))

    if format == "pickle":
        with open(fn, "wb") as f:
            pickle.dump(stats, f)

    elif format == "npz":
        np.savez(
            fn,
            real_names=stats["real_means"].index,
            sample_labels=stats["sample"].columns,
            **stats,
        )

    elif format == "hdf5":
        with pd.HDFStore(fn, "w") as f:
            stats["sample"].to_hdf(f, "sample", format="table")
            stats["real_means"].to_hdf(f, "real_means")
            stats["real_stds"].to_hdf(f, "real_stds")
            for k, v in six.iteritems(stats["value_counts"]):
                v.to_hdf(f, "value_counts/{}".format(k))
            for k in ["n_total", "wt_total", "version", "region_type"]:
                pd.Series([stats[k]]).to_hdf(f, k)

            v = json.dumps(stats.get("version_info", VERSIONS[stats["version"]]))
            pd.Series([v]).to_hdf(f, "version_info")


def load_stats(fn, format=None):
    fn, format = _get_fn_format(fn, format)

    if format == "pickle":
        with open(fn, "rb") as f:
            return pickle.load(f)
    elif format == "npz":
        stats = {k: v[()] for k, v in np.load(fn, allow_pickle=True).items()}

        stats["real_means"] = pd.Series(stats["real_means"], stats["real_names"])
        stats["real_stds"] = pd.Series(stats["real_stds"], stats["real_names"])
        del stats["real_names"]

        stats["sample"] = pd.DataFrame(
            stats["sample"], columns=stats.pop("sample_labels")
        )
        return stats

    elif format == "hdf5":
        stats = {}
        with pd.HDFStore(fn, "r") as f:
            stats["sample"] = f["sample"]
            stats["real_means"] = f["real_means"]
            stats["real_stds"] = f["real_stds"]
            for k in ["n_total", "wt_total", "version", "region_type"]:
                stats[k] = f[k].iloc[0]

            if "version_info" in f:
                stats["version_info"] = json.loads(f["version_info"].iloc[0])
            else:
                stats["version_info"] = deepcopy(VERSIONS[stats["version"]])

            stats["value_counts"] = v = OrderedDict()
            pre = "/value_counts/"
            for k in sorted(f.keys()):
                if k.startswith(pre):
                    v[k[len(pre) :]] = f[k].sort_index()
        return stats
