import os

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
            fn = os.path.join(os.path.dirname(__file__), 'data/regions.h5')
        x = _g[key] = pd.read_hdf(fn, key)
        return x


try:
    from pkg_resources import cleanup_resources
except ImportError:
    pass
else:
    import atexit
    atexit.register(cleanup_resources)
