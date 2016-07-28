import pandas as pd
import six

def save_stats(fn, stats):
    with pd.HDFStore(fn, 'w') as f:
        stats['sample'].to_hdf(f, 'sample', format='table')
        stats['real_means'].to_hdf(f, 'real_means')
        stats['real_stds'].to_hdf(f, 'real_stds')
        for k, v in six.iteritems(stats['value_counts']):
            v.to_hdf(f, 'value_counts/{}'.format(k))
        for k in ['n_total', 'version']:
            pd.Series([stats[k]]).to_hdf(f, k)


def load_stats(fn):
    stats = {}
    with pd.HDFStore(fn, 'r') as f:
        stats['sample'] = f['sample']
        stats['real_means'] = f['real_means']
        stats['real_stds'] = f['real_stds']
        stats['n_total'] = f['n_total'].iloc[0]
        stats['version'] = f['version'].iloc[0]

        stats['value_counts'] = v = {}
        pre = '/value_counts/'
        for k in f.keys():
            if k.startswith(pre):
                v[k[len(pre):]] = f[k]
    return stats
