import argparse
import os

import pandas as pd
import six

from .reader import VERSIONS
from .sort import sort_by_region


def main():
    parser = argparse.ArgumentParser(
        description="Reads American Community Survey Public Use Microdata "
                    "Sample files and featurizes them, as in Flaxman, Wang, "
                    "and Smola (KDD 2015). Needs a preprocessing pass over "
                    "the data to sort it into files by region and to collect "
                    "statistics in order to do one-hot encoding and "
                    "standardization. Currently supports only the 2006-10 "
                    "file used in the original paper.",
    )
    subparsers = parser.add_subparsers()

    sort = subparsers.add_parser(
        'sort', help="Sort the data by region and collect statistics about it.")
    sort.set_defaults(func=do_sort)
    sort.add_argument('out_dir', help='Directory for the sorted features.')
    sort.add_argument('files', nargs='+',
                      help="The original ACS PUMS csv files, from the zip.")
    g = sort.add_mutually_exclusive_group()
    g.add_argument('--voters-only', action='store_true', default=True,
                   help="Only include citizens who are at least 18 years old "
                        "(default).")
    g.add_argument('--all-people', action='store_false', dest='voters_only',
                   help="Include all records from the files.")
    sort.add_argument('--version', choices=VERSIONS, default='2006-10',
                      help="The format of the ACS PUMS files in use; default "
                           "%(default)s.")
    sort.add_argument('--chunksize', type=int, default=10**4, metavar='LINES',
                      help="How much of a CSV file to read at a time; default "
                           "%(default)s.")

    featurize = subparsers.add_parser(
        'featurize', help="Emit features for a given region.")
    featurize.set_defaults(func=do_featurize)

    args = parser.parse_args()
    args.func(args, parser)


def do_sort(args, parser):
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    means, stds, value_counts = sort_by_region(
        args.files, os.path.join(args.out_dir, '{}.csv'),
        voters_only=args.voters_only,
        adj_inc=True, version=args.version, chunksize=args.chunksize)

    fn = os.path.join(args.out_dir, '_stats.h5')
    save_stats(fn, means, stds, value_counts)


def save_stats(fn, means, stds, value_counts):
    if os.path.exists(fn):
        os.remove(fn)
    means.to_hdf(fn, 'means')
    stds.to_hdf(fn, 'stds')
    for k, v in six.iteritems(value_counts):
        v.to_hdf(fn, 'value_counts/{}'.format(k))


def load_stats(fn):
    with pd.HDFStore(fn, 'r') as f:
        means = pd.read_hdf(f, 'means')
        stds = pd.read_hdf(f, 'stds')

        value_counts = {}
        pre = '/value_counts/'
        for k in f.keys():
            if k.startswith(pre):
                value_counts[k[len(pre):]] = pd.read_hdf(f, k)
    return means, stds, value_counts


def do_featurize(args, parser):
    parser.error("Not implemented yet.")
