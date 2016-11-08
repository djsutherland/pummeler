import os

import numpy as np
import pandas as pd


def _stream_or_fn(name):
    try:
        from pkg_resources import resource_stream
        return resource_stream('pummeler', 'data/{}'.format(name))
    except ImportError:
        return os.path.join(os.path.dirname(__file__), name)

def _file_or_fn(name):
    try:
        from pkg_resources import resource_filename
        return resource_filename('pummeler', 'data/{}'.format(name))
    except ImportError:
        return os.path.join(os.path.dirname(__file__), name)


_g = {}
def geocode_data(key):
    try:
        return _g[key]
    except KeyError:
        x = _g[key] = pd.read_hdf(_file_or_fn('regions.h5'), key)
        return x


def centroids_cartesian(year='00'):
    centroids = geocode_data('centroids{}'.format(year))
    theta = centroids.lat * (np.pi / 180)
    phi = centroids.lon * (np.pi / 180)
    r = 6371  # units of km
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return pd.DataFrame(dict(x=x, y=y, z=z))


def election_data(regions='00'):
    if regions == '00':
        p = '2012-by-region.csv.gz'
    else:
        p = '2012-by-region-puma10s.csv.gz'
    return pd.read_csv(_stream_or_fn(p), compression='gzip').set_index('region')


def fod_codes():
    # categories from http://www.census.gov/prod/2012pubs/acs-18.pdf
    return pd.read_csv(_stream_or_fn('fod_cats.csv')).set_index('code')


try:
    from pkg_resources import cleanup_resources
except ImportError:
    pass
else:
    import atexit
    atexit.register(cleanup_resources)
