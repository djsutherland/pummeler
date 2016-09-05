from __future__ import division, print_function
import argparse
from glob import glob
import os

import numpy as np

from .featurize import get_embeddings
from .reader import VERSIONS
from .stats import load_stats, save_stats
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

    ############################################################################
    sort = subparsers.add_parser(
        'sort', help="Sort the data by region and collect statistics about it.")
    sort.set_defaults(func=do_sort)

    io = sort.add_argument_group('Input/output options')
    g = io.add_mutually_exclusive_group(required=True)
    g.add_argument('--zipfile', '-z', help="The original ACS PUMS zip file.")
    g.add_argument('--csv-files', '-c', nargs='+',
                   help="CSV files in ACS PUMS format.")

    io.add_argument('out_dir', help='Directory for the sorted features.')

    io.add_argument('--chunksize', type=int, default=10**5, metavar='LINES',
                      help="How much of a CSV file to read at a time; default "
                           "%(default)s.")
    io.add_argument('--stats-only', action='store_true', default=False,
                    help="Only compute the stats, don't do the sorting.")

    fmt = sort.add_argument_group('Format options')
    g = fmt.add_mutually_exclusive_group()
    g.add_argument('--voters-only', action='store_true', default=True,
                   help="Only include citizens who are at least 18 years old "
                        "(default).")
    g.add_argument('--all-people', action='store_false', dest='voters_only',
                   help="Include all records from the files.")
    fmt.add_argument('--version', choices=VERSIONS, default='2006-10',
                      help="The format of the ACS PUMS files in use; default "
                           "%(default)s.")

    ############################################################################
    featurize = subparsers.add_parser(
        'featurize', help="Emit features for a given region.")
    featurize.set_defaults(func=do_featurize)

    io = featurize.add_argument_group('Input/output options')
    io.add_argument('dir', help="The directory where `pummel sort` put stuff.")
    io.add_argument('outfile', nargs='?',
                    help='Where to put embeddings; default DIR/embeddings.npz.')
    io.add_argument('--chunksize', type=int, default=2**13, metavar='LINES',
                      help="How much of a region to process at a time; default "
                           "%(default)s.")

    emb = featurize.add_argument_group('Embedding options')
    emb.add_argument('--skip-rbf', action='store_true', default=False,
                     help="Skip getting the RBF kernel embedding and only get "
                          "the linear one (much, much faster).")
    emb.add_argument('--n-freqs', type=int, default=2048,
                     help='Number of random frequencies to use (half the '
                          'embedding dimension; default %(default)s).')
    emb.add_argument('--bandwidth', type=float,
                     help='Gaussian kernel bandwidth. Default: choose the '
                          'median distance among the random sample saved in '
                          'the stats file.')
    emb.add_argument('--skip-feats', nargs='+', metavar='FEAT_NAME',
                     help="Don't include some features in the embedding.")

    ############################################################################
    args = parser.parse_args()
    args.func(args, parser)


def do_sort(args, parser):
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    stats = sort_by_region(
        args.zipfile or args.csv_files,
        os.path.join(args.out_dir, 'feats_{}.h5'),
        voters_only=args.voters_only, stats_only=args.stats_only,
        adj_inc=True, version=args.version, chunksize=args.chunksize)
    save_stats(os.path.join(args.out_dir, 'stats.h5'), stats)


def do_featurize(args, parser):
    if args.outfile is None:
        args.outfile = os.path.join(args.dir, 'embeddings.npz')
    stats = load_stats(os.path.join(args.dir, 'stats.h5'))
    files = glob(os.path.join(args.dir, 'feats_*.h5'))
    region_names = [os.path.basename(f)[6:-3] for f in files]
    if args.skip_rbf:
        emb_lin, feature_names = get_embeddings(
            files, stats=stats, chunksize=args.chunksize, skip_rbf=True,
            skip_feats=args.skip_feats)
        np.savez(args.outfile, emb_lin=emb_lin,
                 feature_names=feature_names, region_names=region_names)
    else:
        emb_lin, emb_rff, freqs, bandwidth, feature_names = get_embeddings(
            files, stats=stats, n_freqs=args.n_freqs, bandwidth=args.bandwidth,
            skip_feats=args.skip_feats, chunksize=args.chunksize)
        np.savez(args.outfile,
                 emb_lin=emb_lin, emb_rff=emb_rff,
                 freqs=freqs, bandwidth=bandwidth,
                 feature_names=feature_names, region_names=region_names)
