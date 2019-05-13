from __future__ import division, print_function
import argparse
from functools import partial
from glob import glob
import os
import sys
import traceback

import h5py
import numpy as np
import pandas as pd
import six

from .featurize import get_embeddings
from .misc import get_state_embeddings, get_merged_embeddings
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
                    "standardization.",
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
    fmt.add_argument('--version', choices=sorted(VERSIONS), default='2006-10',
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
    io.add_argument('--force', action='store_true', default=False,
                    help="Force output even if OUTFILE already exists.")
    io.add_argument('--chunksize', type=int, default=2**13, metavar='LINES',
                    help="How much of a region to process at a time; default "
                         "%(default)s.")
    g = io.add_mutually_exclusive_group()
    g.add_argument('--save-compressed', action='store_true', default=False,
                   help="Save embeddings in a compressed .npz. Requires "
                        "enough free space in $TMPDIR, but should compress at "
                        "least emb_lin and especially emb_extra pretty "
                        "reasonably. Default: %(default)s.")
    g.add_argument('--save-uncompressed', action='store_false',
                   dest='save_compressed')
    g = io.add_mutually_exclusive_group()
    g.add_argument('--save-npz',
                   action='store_const', dest='format', const='npz',
                   help="Save outputs in an npz (default).")
    g.add_argument('--save-hdf5',
                   action='store_const', dest='format', const='hdf5',
                   help="Save outputs in an hdf5 file.")
    g.set_defaults(format='npz')

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
    g = emb.add_mutually_exclusive_group()
    g.add_argument('--rff-orthogonal', action='store_true', default=True,
                   help="Use orthogonality in the random features (which "
                        "helps the accuracy of the embedding; default).")
    g.add_argument('--rff-normal', action='store_false', dest='rff_orthogonal',
                   help="Use standard random Fourier features (no "
                        "orthogonality).")
    emb.add_argument('--seed', type=int, default=None,
                     help='Random seed for generating random frequencies. '
                          'Default: none')
    emb.add_argument('--skip-feats', nargs='+', metavar='FEAT_NAME',
                     help="Don't include some features in the embedding.")
    g = emb.add_mutually_exclusive_group()
    g.add_argument('--skip-alloc-flags', action='store_true', default=True,
                   help="Don't include allocation flags (default).")
    g.add_argument('--include-alloc-flags', action='store_false',
                   dest='skip_alloc_flags')
    g = emb.add_mutually_exclusive_group()
    g.add_argument('--do-my-proc', action='store_true', default=False,
                   help="HACK: Do my changes (drop things, cat codes, etc)")
    g.add_argument('--no-my-proc', action='store_false', dest='do_my_proc')
    g = emb.add_mutually_exclusive_group()
    g.add_argument('--do-my-additive', action='store_true', default=False,
                   help="HACK: do additive + some interactions embedding")
    g.add_argument('--no-my-additive', action='store_false', dest='do_my_additive')
    g = emb.add_mutually_exclusive_group()
    g.add_argument('--common-feats', action='store_true', default=False,
                   help="HACK: features common to old and new PUMSes")
    g.add_argument('--no-common-feats', action='store_false', dest='common_feats')
    emb.add_argument('--subsets', metavar='PANDAS_QUERY',
                     help="Comma-separated subsets of the data to calculate "
                          "embeddings for, e.g. "
                          "'SEX == 2 & AGEP > 45, SEX == 2 & PINCP < 20000'.")

    ############################################################################
    export = subparsers.add_parser(
        'export', help="Export features in embeddings.npz as CSV files.")
    export.set_defaults(func=do_export)

    io = export.add_argument_group('Input/output options')
    io.add_argument('dir', help="Where to put the outputs.")
    io.add_argument('infile', nargs='?',
                    help="Location of embeddings created by `pummel feauturize`"
                         "; default DIR/embeddings.npz.")
    io.add_argument('--out-name', metavar='BASE',
                    help="Prefix for embedding output files, so that they "
                         "go e.g. in DIR/BASE_linear.csv. Default to "
                         "the basename of INFILE if it's in DIR or "
                         "otherwise 'embeddings'.")

    ############################################################################
    merge = subparsers.add_parser(
        'merge-features', help="Get embeddings for larger areas from existing "
                               "embeddings.")
    merge.set_defaults(func=do_merge)

    g = merge.add_mutually_exclusive_group(required=True)
    t = partial(g.add_argument, action='store_const', dest='merge_to')
    t('--states', const='states', help="Group into state-level embeddings.")
    t('--merged', const='merged',
      help="Group into regions merged between 2000 and 2010 PUMAs.")

    io = merge.add_argument_group('Input/output options')
    g = io.add_mutually_exclusive_group()
    g.add_argument('--npz', action='store_const', const='npz', dest='format')
    g.add_argument('--hdf5', action='store_const', const='h5', dest='format')

    g = io.add_mutually_exclusive_group()
    g.add_argument('--save-compressed', action='store_true', default=False,
                   help="Save embeddings in a compressed .npz (hdf5 will "
                        "always compress the sparse embeddings). Requires "
                        "enough free space in $TMPDIR, but should compress at "
                        "least emb_lin and especially emb_extra pretty "
                        "reasonably. Default: %(default)s.")
    g.add_argument('--save-uncompressed', action='store_false',
                   dest='save_compressed')

    io.add_argument('infile', help="The existing region embeddings.")
    io.add_argument('outfile', default=None, nargs='?',
                    help="Where to output; default adds _states/_merged to the "
                         "input file name.")

    ############################################################################
    weight_counts = subparsers.add_parser(
        'weight-counts', help="Export total weight per region (approximately "
                              "the number of eligible voters) as a CSV.")
    weight_counts.set_defaults(func=do_weight_counts)

    io = weight_counts.add_argument_group('Input/output options')
    io.add_argument('dir', help="Where the feature files live.")
    io.add_argument('outfile', default=None, nargs='?',
                    help="Where to output; default DIR/weight_counts.csv.")

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
        ext = {'hdf5': 'h5', 'npz': 'npz'}
        args.outfile = os.path.join(
            args.dir, 'embeddings.{}'.format(ext.get(args.format, args.format)))

    if os.path.exists(args.outfile):
        if args.force:
            os.remove(args.outfile)
        else:
            parser.error(("Outfile {} exists. Delete it first or pass --force "
                          "to override it.").format(args.outfile))
    if not os.path.isdir(os.path.dirname(args.outfile)):
        parser.error("Directory {} doesn't exist; is that what you meant?"
                     .format(os.path.dirname(args.outfile)))

    stats = load_stats(os.path.join(args.dir, 'stats.h5'))
    files = glob(os.path.join(args.dir, 'feats_*.h5'))
    region_names = [os.path.basename(f)[6:-3] for f in files]

    res = get_embeddings(
        files=files, stats=stats, chunksize=args.chunksize,
        skip_rbf=args.skip_rbf,
        skip_feats=args.skip_feats, subsets=args.subsets,
        skip_alloc_flags=args.skip_alloc_flags,
        seed=args.seed,
        n_freqs=args.n_freqs, bandwidth=args.bandwidth,
        rff_orthogonal=args.rff_orthogonal,
        do_my_proc=args.do_my_proc, do_my_additive=args.do_my_additive,
        common_feats=args.common_feats)
    res['region_names'] = region_names
    res['subset_queries'] = args.subsets
    _save_embeddings(args.outfile, res, format=args.format,
                     compressed=args.save_compressed)


def _save_embeddings(outfile, res, format='npz', compressed=False):
    try:
        if format == 'npz':
            fn = np.savez_compressed if compressed else np.savez
            fn(outfile, **res)
        elif format == 'hdf5':
            with h5py.File(outfile, 'w') as f:
                for k, v in six.iteritems(res):
                    if k == 'subset_queries' and v is None:
                        continue

                    kwargs = {}

                    # avoid crashing on numpy U types
                    v = np.asanyarray(v)
                    if v.dtype.kind == 'U':
                        kwargs['dtype'] = h5py.special_dtype(vlen=six.text_type)

                    if k in {'emb_lin', 'emb_extra'}:
                        kwargs['compression'] = 'gzip'
                        kwargs['shuffle'] = True

                    d = f.create_dataset(k, shape=v.shape, **kwargs)
                    d[:] = v  # data= doesn't work with unicode apparently...
        else:
            raise ValueError("Unknown output format {!r}".format(format))
    except:
        if sys.stdin.isatty() and sys.stdout.isatty():
            traceback.print_exc()
            print("Dropping you to a shell; result is in `res`, save it "
                  "somewhere else.")
            import IPython
            IPython.embed()
        else:
            raise


def do_export(args, parser):
    if args.infile is None:
        args.infile = os.path.join(args.dir, 'embeddings.npz')

    if args.out_name is None:
        rel = os.path.relpath(args.infile, args.dir)
        if '/' in rel:
            args.out_name = 'embeddings'
        else:
            args.out_name = rel[:-4] if rel.endswith('.npz') else rel
    out_pattern = os.path.join(args.dir, args.out_name + '_{}.csv')

    with np.load(args.infile) as data:
        path = out_pattern.format('linear')
        df = pd.DataFrame(data['emb_lin'])
        df.set_index(data['region_names'], inplace=True)
        df.columns = data['feature_names']
        df.to_csv(path, index_label="region")
        print("Linear embeddings saved in {}".format(path))

        if 'emb_rff' in data:
            path = out_pattern.format('rff')
            df = pd.DataFrame(data['emb_rff'])
            df.set_index(data['region_names'], inplace=True)
            df.to_csv(path, index_label="region")
            print("Fourier embeddings saved in {}".format(path))


def do_merge(args, parser):
    if args.format is None:
        if args.infile.endswith('.npz'):
            args.format = 'npz'
        elif args.infile.endswith('.h5') or args.infile.endswith('.hdf5'):
            args.format = 'hdf5'
        else:
            msg = ("Couldn't infer file format from name '{}'; pass --npz or "
                   "--hdf5.")
            parser.error(msg.format(os.path.basename(args.infile)))

    if args.outfile is None:
        inf = args.infile
        if args.format == 'npz' and args.infile.endswith('.npz'):
            inf = args.infile[:-4]
        elif args.format == 'hdf5':
            if args.infile.endswith('.h5'):
                inf = args.infile[:-3]
            elif args.infile.endswith('.hdf5'):
                inf = args.infile[:-5]
        args.outfile = inf + '_{}.{}'.format(
            args.merge_to, 'npz' if args.format == 'npz' else 'h5')
        print("Output will go in {}".format(args.outfile))

    if os.path.exists(args.outfile):
        parser.error(
            "{} already exists; delete it to force.".format(args.outfile))

    print("Loading...", end='', file=sys.stderr)
    if args.format == 'npz':
        with np.load(args.infile) as f:
            d = dict(**f)
    elif args.format == 'hdf5':
        with h5py.File(args.infile, 'r') as f:
            d = {k: v[()] for k, v in six.iteritems(f)}
    else:
        raise ValueError("confused by args.format {!r}".format(args.format))
    print('done', file=sys.stderr)

    if args.merge_to == 'states':
        embs = get_state_embeddings(d)
    elif args.merge_to == 'merged':
        embs = get_merged_embeddings(d)
    else:
        raise ValueError("confused by args.merge_to {!r}".format(args.merge_to))

    print("Saving...", end='', file=sys.stderr)
    _save_embeddings(args.outfile, embs, format=args.format,
                     compressed=args.save_compressed)
    print('done', file=sys.stderr)


def do_weight_counts(args, parser):
    if args.outfile is None:
        args.outfile = os.path.join(args.dir, 'weight_counts.csv')

    mapping = {}
    for fn in os.listdir(args.dir):
        if fn.startswith('feats_') and fn.endswith('.h5'):
            region = fn[len('feats_'):-len('.h5')]
            with h5py.File(os.path.join(args.dir, fn), 'r') as f:
                mapping[region] = f['total_wt'][()]

    df = pd.DataFrame.from_dict(mapping, orient='index')
    df.columns = ['total_wt']
    df.index.names = ['region']
    df.to_csv(args.outfile)
