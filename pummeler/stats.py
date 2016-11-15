from collections import OrderedDict

import pandas as pd
import six


def save_stats(fn, stats):
    with pd.HDFStore(fn, 'w') as f:
        stats['sample'].to_hdf(f, 'sample', format='table')
        stats['real_means'].to_hdf(f, 'real_means')
        stats['real_stds'].to_hdf(f, 'real_stds')
        for k, v in six.iteritems(stats['value_counts']):
            v.to_hdf(f, 'value_counts/{}'.format(k))
        for k in ['n_total', 'wt_total', 'version']:
            pd.Series([stats[k]]).to_hdf(f, k)


def load_stats(fn):
    stats = {}
    with pd.HDFStore(fn, 'r') as f:
        stats['sample'] = f['sample']
        stats['real_means'] = f['real_means']
        stats['real_stds'] = f['real_stds']
        for k in ['n_total', 'wt_total', 'version']:
            stats[k] = f[k].iloc[0]

        stats['value_counts'] = v = OrderedDict()
        pre = '/value_counts/'
        for k in sorted(f.keys()):
            if k.startswith(pre):
                v[k[len(pre):]] = f[k].sort_index()
    return stats
