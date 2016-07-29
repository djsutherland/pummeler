import os

import numpy as np
import pandas as pd

_g = {}
def geocode_data(key):
    try:
        return _g[key]
    except KeyError:
        try:
            from pkg_resources import resource_filename
            fn = resource_filename('pummeler', 'data/regions.h5')
        except ImportError:
            fn = os.path.join(os.path.dirname(__file__), 'regions.h5')
        x = _g[key] = pd.read_hdf(fn, key)
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


def election_data():
    p = '2012-by-region.csv.gz'
    try:
        from pkg_resources import resource_stream
        f = resource_stream('pummeler', 'data/' + p)
    except ImportError:
        f = os.path.join(os.path.dirname(__file__), p)

    r = pd.read_csv(f, compression='gzip').set_index('region')
    return r


try:
    from pkg_resources import cleanup_resources
except ImportError:
    pass
else:
    import atexit
    atexit.register(cleanup_resources)
