# pummeler

This is a set of utilities for analyzing the American Community Survey's Public Use Microdata Sample files (ACS PUMS), mostly following

> Flaxman, Wang, and Smola. Who Supported Obama in 2012? Ecological Inference through Distribution Regression. KDD 2015.
> ([official](http://dx.doi.org/10.1145/2783258.2783300), [from author](http://sethrf.com/files/ecological.pdf))


## Usage

### Installation

`pip install pummeler`; it will then be available via `import pummler` and install a `pummel` script.

If you prefer, you can also check out the source directory, which should work as long as you put the `pummel` directory on your `sys.path` (or start Python from the root of the checkout). In that case you should use the `pummel` script at the top level of the checkout.


### Getting the census data

First, download the data from the Census site. You probably want the "csv_pus.zip" file from whatever distribution you're using. The currently supported options are:

- [2006-10](http://www2.census.gov/programs-surveys/acs/data/pums/2010/5-Year/csv_pus.zip) (2.1 GB); uses 2000 PUMAs.
- [2007-11](http://www2.census.gov/programs-surveys/acs/data/pums/2011/5-Year/csv_pus.zip) (2.1 GB); uses 2000 PUMAs.
- The 2012-14 subset of the [2010-14 file](https://www2.census.gov/programs-surveys/acs/data/pums/2014/5-Year/csv_pus.zip) (2.3 GB); this is the subset using 2010 PUMAs. (Pass `--version 2010-14_12-14` to `pummel`.)
- [2015](https://www2.census.gov/programs-surveys/acs/data/pums/2015/1-Year/csv_pus.zip) (595 MB); uses 2010 PUMAs.
- The 2012-15 subset of the [2012-15 file](https://www2.census.gov/programs-surveys/acs/data/pums/2015/5-Year/csv_pus.zip) (2.4GB); this is the subset using 2010 PUMAs. (Pass `--version 2011-15_12-15` to `pummel`.)
- A gross manual merger of the 2012-14 and 15 data via [`make_12to15.py`](make_12to15.py); use the proper option above instead (which was only made available after we wanted it).

It's relatively easy to add support for new versions; see the `VERSIONS` dictionary in [`pummeler.reader`](pummeler/reader.py).

### Picking regions of analysis

Election results are generally reported by counties; PUMS data are in their own special [Public Use Microdata Areas](https://www.census.gov/geo/reference/puma.html), which are related but not the same. This module ships with regions that merge all overlapping blockgroups / counties, found with [the MABLE/Geocorr tool](http://mcdc2.missouri.edu/websas/geocorr12.html), in Pandas dataframes stored in `pummeler/data/regions.h5`.

Regions are named like `AL_00_01`, which means Alabama's region number 01 in the 2000 geography, or `WY_10_02`, which is Wyoming's second region in the 2010 geography. There are also "superregions" which merge 2000 and 2010 geographies, named like `PA_merged_03`.

**Note:** Alaskan electoral districts are weird. For now, I just lumped all of Alaska into one region.

This was done in the Jupyter notebook [`notebooks/get regions.ipynb`](notebooks/get%20regions.ipynb). Centroids are calculated in [`notebooks/region centroids.ipynb`](notebooks/region%20centroids.ipynb), using a shapefile for counties [from here](https://geonet.esri.com/thread/24614).

**TODO:** Could switch to precinct-level results, which should end up with more regions in the end. 2012 results are available [here](http://projects.iq.harvard.edu/eda/data), including shapefiles if you go into the state-by-state section, so it shouldn't be *too* much work there. I haven't found national precinct-level results for the 2016 election yet, but maybe somebody's done it.


### Preprocessing

First, we need to sort the features by region, and collect statistics about them so we can do the featurization later.

Run `pummel sort --version 2006-10 -z csv_pus.zip SORT_DIR`. (A few extra options are shown if you pass `--help`.) This will:

- Make a bunch of files in `SORT_DIR` like `feats_AL_00_01.h5`, which contain basically the original features (except with the `ADJINC` adjustment applied to fields that need it to account for inflation) grouped by region. These are stored in HDF5 format with pandas, because it's much faster and takes less disk space than CSVs.

- Makes a file `SORT_DIR/stats.h5` containing means and standard deviations of the real-valued features, counts of the different values for the categorical features, and a random sample of all the features.

This will take a while (~15 minutes on my fast-ish laptop) and produce about 4GB of temp data (for the 2006-10 files). Luckily you should only need to do it once per ACS file.


### Featurization

Run `pummel featurize SORT_DIR`. (Again, you have a couple of options shown by `--help`.) This will get both linear embeddings (i.e. means) and random Fourier feature embeddings for each region, saving the output in `SORT_DIR/embeddings.npz`.

You can also get features for demographic subsets with e.g. `--subsets 'SEX == 2 & AGEP > 45, SEX == 1 & PINCP < 20000'`.

**NOTE:** As it turns out, with this featurization, linear embeddings seem to be comparable to random Fourier feature embeddings. You can save yourself a bunch of time and the world a smidgen of global warming if you skip them with `--skip-rbf`.

On my laptop (with a quad-core Haswell i7), doing it with random Fourier features takes about an hour; the only-linear version takes about ten minutes. Make sure you're using a numpy linked to a fast multithreaded BLAS (like MKL or OpenBLAS; the easiest way to do this is to use the [Anaconda](https://www.continuum.io/downloads) Python distribution, which includes MKL by default); otherwise, this step will be much slower.

If it's using too much memory, decrease `--chunksize`.

The original paper used Fastfood transforms instead of the default random Fourier features used here, which with a good implementation will be faster. I'm not currently aware of a high-quality, easily-available Python-friendly implementation. A GPU implementation of regular random Fourier features could also help.

`SORT_DIR/embeddings.npz`, which you can load with `np.load`, will then have:

 - `emb_lin`: the `n_regions x n_feats` array of feature means.
 - `emb_rff`: the `n_regions x (2 * n_freq)` array of random Fourier feature embeddings.
 - `region_names`: the names corresponding to the first axis of the embeddings.
 - `feature_names`: the names for each used feature.
 - `freqs`: the `n_feats x n_freq` array of random frequencies for the random Fourier features.
 - `bandwidth`: the bandwidth used for selecting the `freqs`.

 (If you did `--skip-rbf`, `emb_rff`, `freqs`, and `bandwidth` won't be present.)


### Getting the election data

This package includes results derived from [`huffpostdata/election-2012-results`](https://github.com/huffpostdata/election-2012-results), in `pummeler/data/2012-by-region.csv.gz`. That data was created in [`notebooks/election data by region.ipynb`](notebooks/election%20data%20by%20region.ipynb).

There doesn't seem to be a good publicly-available county-level election results resource for years prior to 2012. If you get some, follow that notebook to get results in a similar format. (Your might have an institutional subscription to CQ Press's election data, for example. That source, though, doesn't use FIPS codes, so it'll be a little more annoying to line up; I might do that at some point.)

**TODO:** add 2016 election data.

### Analysis

For a basic replication of the model from the paper, see [`notebooks/analyze.ipynb`](notebooks/analyze.ipynb).
