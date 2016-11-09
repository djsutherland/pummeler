from __future__ import division, print_function
from collections import defaultdict
import itertools
import sys

import numpy as np
import pandas as pd
import progressbar as pb
from scipy.linalg import qr
from sklearn.metrics.pairwise import euclidean_distances
import six
from six.moves import xrange

from .data import fod_codes
from .reader import VERSIONS


_cache_needs_nan = {}
def _needs_nan(c, stats):
    try:
        return _cache_needs_nan[c]
    except KeyError:
        vc_sum = stats['value_counts'][c].sum()
        _cache_needs_nan[c] = b = vc_sum < stats['n_total']
        return b


def get_dummies(df, stats, num_feats=None, ret_df=True, skip_feats=None,
                dtype=np.float64, out=None):
    '''
    Gets features for the person records in `df`: standardizes the real-valued
    features, and does one-hot encoding for the discrete ones. Skip any
    features in skip_feats.
    '''
    info = VERSIONS[stats['version']]
    skip_feats = set() if skip_feats is None else set(skip_feats)
    if num_feats is None:
        num_feats = _num_feats(stats, skip_feats=skip_feats)

    if out is None:
        out = np.empty((df.shape[0], num_feats), dtype=dtype)
    else:
        assert out.shape == (df.shape[0], num_feats)

    real_feats = [f for f in info['real_feats'] if f not in skip_feats]

    reals = out[:, :len(real_feats)]
    reals[:] = df[real_feats]
    reals[:] -= stats['real_means'][real_feats]
    reals[:] /= stats['real_stds'][real_feats]
    reals[np.isnan(reals)] = 0
    if ret_df:
        feat_names = list(real_feats)
    start_col = len(real_feats)

    for k, vc in six.iteritems(stats['value_counts']):
        if k in skip_feats:
            continue

        needs_nan = _needs_nan(k, stats)
        n_codes = len(vc) + int(needs_nan)
        _get_dummies(df[k], vc, with_nan=needs_nan,
                     out=out[:, start_col:start_col+n_codes])

        if ret_df:
            feat_names += ['{}_{}'.format(k, v) for v in vc.index]
            if needs_nan:
                feat_names.append('{}_nan'.format(k))

        start_col += n_codes
    assert start_col == num_feats

    if ret_df:
        return pd.DataFrame(out, index=df.index, columns=feat_names)
    else:
        return out

def _get_dummies(col, vc, with_nan, out=None, dtype=np.float64):
    c = pd.Categorical(col, categories=vc.index).codes
    n_codes = len(vc)
    if with_nan:
        c = c.copy()
        c[c == -1] = n_codes
        n_codes += 1

    if out is None:
        out = np.empty((col.shape[0], n_codes), dtype=dtype)
    np.eye(n_codes).take(c, axis=0, out=out)
    return out


def _num_feats(stats, skip_feats=None):
    skip_feats = set() if skip_feats is None else set(skip_feats)
    n_total = stats['n_total']

    n = len(set(stats['real_means'].index) - skip_feats)
    for k, v in six.iteritems(stats['value_counts']):
        if k not in skip_feats:
            n += v.size + (1 if v.sum() < n_total else 0)
    return n


################################################################################
### Embeddings

def linear_embedding(feats, wts, out=None):
    '''
    Gets the linear kernel embedding (which is just the weighted mean) for
    dummy features `feats`, with sample weighting `wts`.
    '''
    if out is None:
        out = np.empty((feats.shape[1], wts.shape[0]))
    np.dot(feats.T, wts.T, out=out)
    w = wts.sum(axis=1)
    nz = w != 0
    out[:, nz] /= w[np.newaxis, nz]
    return out


def rff_embedding(feats, wts, freqs, out=None):
    '''
    Gets the random Fourier feature embedding for dummy features `feats`,
    with sample weighting `wts`.
    '''
    D = freqs.shape[1]
    if out is None:
        out = np.empty((2 * D, wts.shape[0]))

    angles = np.dot(feats, freqs)
    sin_angles = np.sin(angles)  # TODO: could use MKL sincos for this
    cos_angles = np.cos(angles, out=angles)

    np.dot(sin_angles.T, wts.T, out=out[:D])
    np.dot(cos_angles.T, wts.T, out=out[D:])
    w = wts.sum(axis=1)
    nz = w != 0
    out[:, nz] /= w[np.newaxis, nz] * np.sqrt(D)
    return out


def pick_rff_freqs(n_freqs, bandwidth, seed=None, n_feats=None,
                   orthogonal=True, stats=None, skip_feats=None):
    '''
    Sets up sampling with random Fourier features corresponding to a Gaussian
    kernel with the given bandwidth, with an embedding dimension of `2*n_freqs`.

    Either pass n_feats, or pass stats (and maybe skip_feats) to compute it.

    If orthogonal, uses Orthogonal Random Features:
      https://arxiv.org/abs/1610.09072
    '''
    if n_feats is None:
        n_feats = _num_feats(stats, skip_feats=skip_feats)
    rs = np.random.mtrand._rand if seed is None else np.random.RandomState(seed)

    if not orthogonal or n_feats == 1:  # ORF doesn't do anything for d=1
        return rs.normal(0, 1 / bandwidth, size=(n_feats, n_freqs))

    n_reps = int(np.ceil(n_freqs / n_feats))
    freqs = np.empty((n_feats, n_freqs))
    for i in range(n_reps):
        Q, _ = qr(rs.normal(0, 1, size=(n_feats, n_feats)), overwrite_a=True)
        if i < n_reps - 1:
            freqs[:, i*n_feats:(i+1)*n_feats] = Q.T
        else:
            freqs[:, i*n_feats:] = Q[:n_freqs - i*n_feats].T

    S = rs.chisquare(n_feats, size=n_freqs)
    np.sqrt(S, out=S)
    S /= bandwidth
    freqs *= S[np.newaxis, :]
    return freqs


def pick_gaussian_bandwidth(stats, skip_feats=None):
    '''
    Finds the median distance between features from the random sample saved
    in stats.
    '''
    return _get_median(get_dummies(
        stats['sample'], stats, ret_df=False, skip_feats=skip_feats))


def _get_median(samp):
    D2 = euclidean_distances(samp, squared=True)
    return np.sqrt(np.median(D2[np.triu_indices_from(D2, k=1)]))


################################################################################

def get_embeddings(files, stats, n_freqs=2048, freqs=None, bandwidth=None,
                   chunksize=2**13, skip_rbf=False, skip_feats=None, seed=None,
                   rff_orthogonal=True, subsets=None,
                   squeeze_queries=True, skip_alloc_flags=True,
                   do_my_proc=False, do_my_additive=False):
    skip_feats = set() if skip_feats is None else set(skip_feats)
    if skip_alloc_flags:
        skip_feats.update(VERSIONS[stats['version']]['alloc_flags'])

    if do_my_proc or do_my_additive:
        skip_feats.update(_my_proc_setup(stats))

    if do_my_additive:
        do_my_proc = True
        m = my_additive_setup(stats, skip_feats, seed)

    n_feats = _num_feats(stats, skip_feats=skip_feats)
    feat_names = None

    if not skip_rbf:
        if freqs is None:
            if bandwidth is None:
                print("Picking bandwidth by median heuristic...",
                      file=sys.stderr, end='')
                bandwidth = pick_gaussian_bandwidth(
                        stats, skip_feats=skip_feats)
                print("picked {}".format(bandwidth), file=sys.stderr)
            freqs = pick_rff_freqs(
                n_freqs, bandwidth, seed=seed, n_feats=n_feats,
                orthogonal=rff_orthogonal)
        else:
            n_freqs = freqs.shape[1]


    if subsets is None:
        subsets = 'PWGTP > 0'
    n_subsets = subsets.rstrip()[:-1].count(',') + 1  # allow trailing comma
    if n_subsets == 1:
        subsets += ','  # make sure eval returns a matrix
    # This should work for anything we want, I think

    emb_lin = np.empty((len(files), n_feats, n_subsets))
    if not skip_rbf:
        emb_rff = np.empty((len(files), 2 * n_freqs, n_subsets))
    if do_my_additive:
        emb_extra = np.empty((len(files), m.n_extra, n_subsets))
    region_weights = np.empty((len(files), n_subsets))

    bar = pb.ProgressBar(max_value=stats['n_total'])
    bar.start()
    read = 0
    dummies = np.empty((chunksize, n_feats))
    for file_idx, file in enumerate(files):
        emb_lin_pieces = []
        if not skip_rbf:
            emb_rff_pieces = []
        if do_my_additive:
            emb_extra_pieces = []
        weights = []
        total_weights = 0
        for c in pd.read_hdf(file, chunksize=chunksize):
            read += c.shape[0]
            bar.update(read)

            if do_my_proc:
                _my_proc_chunk(c, skip_feats=skip_feats)

            hacked = False
            if c.shape[0] == 1:
                # gross pandas bug in this case
                c = pd.concat([c, c])
                hacked = True

            which = c.eval(subsets).astype(bool)
            if hacked:
                c = c.iloc[:1]
                which = which[:, :1]

            keep = which.any(axis=0)
            c = c.loc[keep]
            which = which[:, keep]
            if not c.shape[0]:
                continue

            feats = dummies[:c.shape[0], :]

            df = get_dummies(c, stats, num_feats=n_feats, skip_feats=skip_feats,
                             ret_df=feat_names is None, out=feats)
            if feat_names is None:
                feat_names = list(df.columns)
                feat_identities = [n.split('_', 1)[0] for n in feat_names]

                feat_locs = defaultdict(list)
                for i, s in enumerate(feat_names):
                    var = s.split('_', 1)[0]
                    feat_locs[var].append(i)

            wts = np.tile(c.PWGTP, (n_subsets, 1))
            for i, w in enumerate(which):
                wts[i, ~w] = 0

            emb_lin_pieces.append(linear_embedding(feats, wts))
            if not skip_rbf:
                emb_rff_pieces.append(rff_embedding(feats, wts, freqs))

            if do_my_additive:
                emb_extra_pieces.append(my_additive_extras(
                    feats, wts, m, feat_locs))

            ws = wts.sum(axis=1)
            weights.append(ws)
            total_weights += ws

        ratios = []
        for ws in weights:
            ratio = np.array(ws, dtype=float, copy=True)
            nz = total_weights != 0
            ratio[nz] /= total_weights[nz]
            ratios.append(ratio)

        emb_lin[file_idx] = 0
        for rs, l in zip(ratios, emb_lin_pieces):
            emb_lin[file_idx] += l * rs

        if not skip_rbf:
            emb_rff[file_idx] = 0
            for rs, r in zip(ratios, emb_rff_pieces):
                emb_rff[file_idx] += r * rs

        if do_my_additive:
            emb_extra[file_idx] = 0
            for rs, p in zip(ratios, emb_extra_pieces):
                emb_extra[file_idx] += p * rs

        region_weights[file_idx] = total_weights
    bar.finish()

    if squeeze_queries and n_subsets == 1:
        emb_lin = emb_lin[:, :, 0]
        if not skip_rbf:
            emb_rff = emb_rff[:, :, 0]
        if do_my_additive:
            emb_extra = emb_extra[:, :, 0]
        region_weights = region_weights[:, 0]

    # figure out which levels we'd want to drop for the sake of
    # killing perfect colinearity in dummy features
    def _keeps(identities):
        _, ids = np.unique(identities, return_inverse=True)
        starts = np.diff(ids).nonzero()[0]
        ends = np.r_[starts[1:], len(identities)]
        keeps = np.ones(len(identities), dtype=bool)
        keeps[starts[ends - starts > 1]] = False
        return keeps

    keep_multilevels = _keeps(feat_identities)
    if do_my_additive:
        # only want to drop this for discrete-discrete features;
        extra_keep_multilevels = _keeps(m.extra_identities)
        for i in (~extra_keep_multilevels).nonzero()[0]:
            if m.extra_names[i].endswith('_rff'):
                extra_keep_multilevels[i] = True

    ret = {
        'emb_lin': emb_lin,
        'region_weights': region_weights,
        'feature_names': feat_names,
        'feature_identities': feat_identities,
        'keep_multilevels': keep_multilevels,
    }
    if not skip_rbf:
        ret['emb_rff'] = emb_rff
        ret['freqs'] = freqs
        ret['bandwidth'] = bandwidth
    if do_my_additive:
        ret['emb_extra'] = emb_extra
        ret['extra_names'] = m.extra_names
        ret['extra_identities'] = m.extra_identities
        ret['rff_reals'] = m.rff_reals
        ret['rff_pairs'] = m.rff_pairs
        ret['rff_discrete_pairs'] = m.rff_discrete_pairs
        ret['discrete_pairs'] = m.discrete_pairs
        ret['one_bws'] = m.one_bws
        ret['pair_bws'] = m.pair_bws
        ret['one_freqs'] = m.one_freqs
        ret['pair_freqs'] = m.pair_freqs
        ret['extra_keep_multilevels'] = extra_keep_multilevels

    return ret


################################################################################

_naics_cat = {
    '11': 'Agriculture',
    '21': 'Extraction',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '3M': 'Manufacturing',
    '42': 'Wholesale',
    '44': 'Retail',
    '45': 'Retail',
    '4M': 'Retail',
    '48': 'Transportation',
    '49': 'Transportation',
    '51': 'Information',
    '52': 'Financial Services',
    '53': 'Rental and Leasing',
    '54': 'Science and Technical',
    '55': 'Management',
    '56': 'Misc Professions',
    '61': 'Education',
    '62': 'Medical',
    '71': 'Entertainment',
    '72': 'Tourism',
    '81': 'Services',
    '92': 'Administration',
    '99': 'Unemployed',
}
def naics_cat(s):
    return _naics_cat[str(s)[:2]]

occ_cats = [
    (   0, "N/A"),
    (   1, "Management, Business, Science, and Arts Occupations"),
    ( 500, "Business Operations Specialists"),
    ( 800, "Financial Specialists"),
    (1000, "Computer and Mathematical Occupations"),
    (1300, "Architecture and Engineering Occupations"),
    (1600, "Life, Physical, and Social Science Occupations"),
    (2000, "Community and Social Services Occupations"),
    (2100, "Legal Occupations"),
    (2200, "Education, Training, and Library Occupations"),
    (2600, "Arts, Design, Entertainment, Sports, and Media Occupations"),
    (3000, "Healthcare Practitioners and Technical Occupations"),
    (3600, "Healthcare Support Occupations"),
    (3700, "Protective Service Occupations"),
    (4000, "Food Preparation and Serving Occupations"),
    (4200, "Building and Grounds Cleaning and Maintenance Occupations"),
    (4300, "Personal Care and Service Occupations"),
    (4700, "Sales and Related Occupations"),
    (5000, "Office and Administrative Support Occupations"),
    (6000, "Farming, Fishing, and Forestry Occupations"),
    (6200, "Construction and Extraction Occupations"),
    (6800, "Extraction Workers"),
    (7000, "Installation, Maintenance, and Repair Workers"),
    (7700, "Production Occupations"),
    (9000, "Transportation and Material Moving Occupations"),
    (9800, "Military Specific Occupations"),
]
occ_codes, occ_names = zip(*occ_cats)
occ_codes = np.array(occ_codes)
def occ_cat(x):
    return occ_names[occ_codes.searchsorted(float(x), side='right') - 1]

fod_cats = {k: v for k, v in fod_codes().cat_code.iteritems()}


def _my_proc_setup(stats):
    assert stats['version'] in {'2010-14_12-14', '2012-15_manual', '2015'}

    # inc_feats = set()
    # inc = lambda a: inc_feats.update(a.split())
    inc = lambda a: None  # here for documentation only
    skip_feats = set()
    skip = lambda a: skip_feats.update(a.split())

    # basic info: keep AGEP, SEX
    inc('AGEP SEX')
    skip('QTRBIR')

    # drop stuff about relationships to the reference person
    skip('RELP OC RC SFN SFR')

    # citizenship
    inc('CIT DECADE NATIVITY NOP CITWP')
    skip('YOEP')  # DECADE should be good enough

    # employment:
    inc('COW ESR WKHP WKL WKW WRK')
    skip('INDP SOCP')  # TODO: categories here
    inc('NAICSP OCCP')  # will be recoded below
    inc('ESP')  # for kids: are parents employed?
    skip('NWAB NWAV NWLA NWLK NWRE')

    # commute
    inc('JWDP')  # time left for work
    skip('JWAP')  # time arrived at work; fairly redundant with JWDP
    inc('JWMNP')  # travel time
    inc('JWRIP')  # carpooling
    skip('DRIVESP')  # 1/JWRIP
    inc('JWTR')  # commute method
    skip('POWPUMA POWSP')

    # income
    inc('INTP OIP PAP RETP SEMP SSIP SSP WAGP PERNP PINCP POVPIP')
    # NOTE: POVPIP sharply discontinuous, should discretize more

    # kids
    inc('FER')  # woman who gave birth in last year
    inc('PAOC')  # woman with kids' ages
    inc('GCL GCM GCR')  # grandparents living with grandkids

    # education
    inc('SCH SCHG SCHL SCIENGP SCIENGRLP')
    inc('FOD1P'); skip('FOD2P')  # recoded into categories below

    # disability
    inc('DDRS DEAR DEYE DOUT DPHY DREM DRAT DRATX DIS')

    # marital
    inc('MSP MARHT MARHD MARHM MARHW MARHYP')
    skip('MAR')  # superceded by MSP

    # language
    inc('ENG LANX LANP')

    # health insurance
    inc('HICOV PRIVCOV PUBCOV')
    skip('HINS1 HINS2 HINS3 HINS4 HINS5 HINS6 HINS7')

    # migration
    inc('MIG WAOB')  # lived here a year ago, world area of birth
    skip('MIGPUMA MIGSP')
    skip('POBP')

    # military
    inc('MIL VPS')
    skip('MLPA MLPB MLPCD MLPE MLPFG MLPH MLPI MLPJ MLPK')

    # ancestry
    inc('HISP')  # 24 levels, area of hispanic origin
    inc('RAC1P RAC2P RAC3P')  # lots of levels here
    inc('RACAIAN RACASN RACBLK RACNHPI RACSOR RACWHT RACNUM')
    skip('ANC')
    inc('ANC1P ANC2P')

    # modify stats
    vc = stats['value_counts']
    vc['NAICSP'] = vc['NAICSP'].groupby(naics_cat).sum()
    vc['OCCP'] = vc['OCCP'].groupby(occ_cat).sum()
    vc['FOD1P'] = vc['FOD1P'].groupby(fod_cats).sum()
    vc['HASDEGREE'] = vc['SCHL'].groupby(lambda x: int(x >= 20)).sum()
    vc['ANYHISP'] = vc['HISP'].groupby(lambda x: int(x == 1)).sum()
    vc['ETHNICITY'] = pd.Series(
        [stats['n_total'] - 5] + [1] * 5,
        index=['hispanic', 'white', 'black', 'amerindian', 'asian',
               'other/biracial'])
    # obviously the value counts are a lie, but we don't actually use them

    _my_proc_chunk(stats['sample'])

    stats['_added_discrete'] = {'ANYHISP', 'HASDEGREE', 'ETHNICITY'}
    return skip_feats


_ethnicity_map = {
        1: 'white', 2: 'black', 3: 'amerindian', 4: 'amerindian',
        5: 'amerindian', 6: 'asian', 7: 'amerindian', 8: 'other/biracial',
        9: 'other/biracial', 'hispanic': 'hispanic'}

def _my_proc_chunk(df, skip_feats=set()):
    # get NAICS category
    if 'NAICPS' not in skip_feats:
        df['NAICSP'] = df.NAICSP.map(naics_cat, na_action='ignore')

    # get OCC categories
    if 'OCCP' not in skip_feats:
        df['OCCP'] = df.OCCP.astype(float).map(occ_cat, na_action='ignore')

    # get field of degree categories
    # was averaging these before, but that's a little complicated in this
    # code structure, so whatever
    if 'FOD1P' not in skip_feats:
        df['FOD1P'] = df.FOD1P.map(fod_cats, na_action='ignore')
    if 'FOD2P' not in skip_feats:
        df['FOD2P'] = df.FOD2P.map(fod_cats, na_action='ignore')

    if 'ANYHISP' not in skip_feats:
        df['ANYHISP'] = (df.HISP > 1).astype(int)
    if 'HASDEGREE' not in skip_feats:
        df['HASDEGREE'] = (df.SCHL >= 20).astype(int)

    if 'ETHNICITY' not in skip_feats:
        # ETHNICITY is a recode
        df['ETHNICITY'] = df.RAC1P.where(df.HISP == 1,
                                         'hispanic').map(_ethnicity_map)


# Other changes that need to be done in sort (:|):
# income recoding (log-scale, percentages for categories?)
# povpip recoding (0-500 can be real, but 501 needs to be discrete)


def my_additive_setup(stats, skip_feats, seed):
    class Model(object):
        pass
    m = Model()
    m.stats = stats

    # We're doing:
    #   Mean (linear) features for everything.
    #   Additional RFF features for:
    m.rff_reals = ['AGEP', 'PINCP', 'WKHP']
    m.one_n_freqs = 128
    m.pair_n_freqs = 256
    #   Interaction features between all pairs of those and
    m.discretes = ["SEX", "ETHNICITY", "HASDEGREE"]

    for d in m.discretes:
        assert d not in skip_feats

    m.rff_pairs = list(itertools.combinations(m.rff_reals, 2))
    m.rff_discrete_pairs = [
        (r, d) for r in m.rff_reals for d in m.discretes]
    m.discrete_pairs = list(itertools.combinations(m.discretes, 2))

    vcs = stats['value_counts']
    _levels = {}
    def levels(d):
        if d not in _levels:
            _levels[d] = l = ['{}'.format(v) for v in vcs[d].index]
            if _needs_nan(d, stats):
                l.append('nan')
        return _levels[d]

    m.extra_names, m.extra_identities = zip(*(
        [('{}_{}{}'.format(r, sc, i), '{}_rff'.format(r))
         for r in m.rff_reals
         for sc in ['sin', 'cos']
         for i in xrange(m.one_n_freqs)] +
        [('{}_{}_{}{}'.format(d1, d2, sc, i), '{}_{}_rff'.format(d1, d2))
         for d1, d2 in m.rff_pairs
         for sc in ['sin', 'cos']
         for i in xrange(m.pair_n_freqs)] +
        [('{}_{}_{}_{}{}'.format(d, r, v, sc, i), '{}_{}_rff'.format(d, r))
         for r, d in m.rff_discrete_pairs
         for sc in ['sin', 'cos']
         for i in xrange(m.one_n_freqs)
         for v in levels(d)] +
        [('{}_{}_{}_{}'.format(d1, d2, v1, v2), '{}_{}'.format(d1, d2))
         for d1, d2 in m.discrete_pairs
         for v1 in levels(d1)
         for v2 in levels(d2)]))
    m.n_extra = len(m.extra_names)

    samp = np.array(stats['sample'][m.rff_reals], copy=True)
    samp -= stats['real_means'][m.rff_reals]
    samp /= stats['real_stds'][m.rff_reals]
    samp[np.isnan(samp)] = 0

    m.one_bws = [
        _get_median(samp[:, [i]]) for i in range(len(m.rff_reals))]
    m.pair_bws = [
        _get_median(samp[:, [i, j]])
        for i, j in itertools.combinations(range(len(m.rff_reals)), 2)]

    rs = (np.random.mtrand._rand if seed is None
          else np.random.RandomState(seed + 1))
    m.one_freqs = [
        pick_rff_freqs(m.one_n_freqs, bandwidth=bw, n_feats=1,
                       seed=rs.randint(2**23))
        for bw in m.one_bws]
    m.one_freqs_d = {n: f for n, f in zip(m.rff_reals, m.one_freqs)}
    m.pair_freqs = [
        pick_rff_freqs(m.pair_n_freqs, bandwidth=bw, n_feats=2,
                       seed=rs.randint(2**23))
        for bw in m.pair_bws]

    return m


def my_additive_extras(feats, wts, m, feat_locs, out=None):
    pos = 0
    n_subs, n = wts.shape

    if out is None:
        out = np.empty((m.n_extra, n_subs))

    # real-only features
    step = 2 * m.one_n_freqs
    for f, freqs in zip(m.rff_reals, m.one_freqs):
        rff_embedding(feats[:, feat_locs[f]], wts, freqs, out=out[pos:pos+step])
        pos += step

    # real pair features
    step = 2 * m.pair_n_freqs
    for (f1, f2), freqs in zip(m.rff_pairs, m.pair_freqs):
        rff_embedding(feats[:, feat_locs[f1]+feat_locs[f2]], wts, freqs,
                      out=out[pos:pos+step])
        pos += step

    # # dummies for our discrete features; helpful for later
    # print('dummies')
    # dummies = {}
    # for d in m.discretes:
    #     dummies[d] = _get_dummies(
    #         feats[d], m.stats['value_counts'][d],
    #         with_nan=_needs_nan(d, m.stats), dtype=np.uint8)

    # real-discrete pair features
    # want to make weight subsets that only hit the relevant feats
    for fr, fd in m.rff_discrete_pairs:
        d = feats[:, feat_locs[fd]]
        l = d.shape[1]
        freqs = m.one_freqs_d[fr]
        step = 2 * m.one_n_freqs * l

        d_wts = d.T[np.newaxis, :, :] * wts[:, np.newaxis, :]  # n_subs, l, n
        d_wts = d_wts.reshape(l * n_subs, n)
        tmp_out = rff_embedding(feats[:, feat_locs[fr]], d_wts, freqs)  # D, l * n_subs
        out[pos:pos + step] = tmp_out.reshape(-1, n_subs)
        pos += step

    # discrete interactions
    for f1, f2 in m.discrete_pairs:
        d1 = feats[:, feat_locs[f1]]
        d2 = feats[:, feat_locs[f2]]
        l1 = d1.shape[1]
        l2 = d2.shape[1]
        step = l1 * l2
        d = (d1[:, :, np.newaxis] * d2[:, np.newaxis, :]).reshape(n, step)
        linear_embedding(d, wts, out=out[pos:pos+step])
        pos += step

    assert pos == m.n_extra
    return out
