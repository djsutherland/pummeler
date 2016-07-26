# pummeler

This is a set of utilities for analyzing the American Community Survey's Public Use Microdata Survey files (ACS PUMS), mostly following

> Flaxman, Wang, and Smola. Who Supported Obama in 2012? Ecological Inference through Distribution Regression. KDD 2015.
> ([official](http://dx.doi.org/10.1145/2783258.2783300), [from author](http://sethrf.com/files/ecological.pdf))

The package currently only supports the 2006-10 5-year estimates, but more will be added.


## Usage

### Installation

`pip install pummeler` should work soon, once it's done enough to have an actual release; it will then be available via `import pummler` and install a `pummel` script.

If you prefer, you can also check out the source directory, which should work as long as you put the `pummel` directory on your `sys.path` (or start Python from the root of the checkout). In that case you should use the `pummel` script at the top level of the checkout.


### Getting the census data

First, download the data from the Census site. You probably want the "csv_pus.zip" file, [here](http://www2.census.gov/programs-surveys/acs/data/pums/2010/5-Year/csv_pus.zip) for the 2006-10 data (2.1 GB).

Extract it. I'll assume the CSV files ended up in the path `PUMS_SOURCE`.


### Getting the election data

There doesn't seem to be a good publicly-available county-level election results resource for years prior to 2012.

We currently use results from CQ Press. **TODO: how to get**

**TODO:** update to support HuffPo 2012 results file or similar.


### Picking regions of analysis

Election results are reported by counties; PUMS data are in census blockgroups, which are related but not the same. This module ships with regions that merge all overlapping blockgroups / counties, found with [the MABLE/Geocorr tool](http://mcdc2.missouri.edu/websas/geocorr12.html), in Pandas dataframes stored in `data/regions.h5`.

Regions are named like `AL_00_01`, which means Alabama's region number 01 in the 2000 geography. If you use 2010 geographies (i.e. you're using ACS data of vintage after 2011), those regions are named `AL_10_01`.

**Note:** Alaska electoral districts are weird. I just lumped all of Alaska into one region.

This was done in the Jupyter notebook `notebooks/get electoral regions.ipynb`.

**TODO:** add notebook, accounting for election data issues


### Preprocessing

First, we need to sort the features by region and collect statistics about them so we can do the featurization.

Run `pummel sort SORT_DIR PUMS_SOURCE/ss*.csv`. (A few extra options are shown if you pass `--help`.) This will:

- Make a bunch of files in `OUT_DIR` like `AL_00_01.csv`, which contain basically the original features (except with the `ADJINC` adjustment applied to fields that need it to account for inflation) grouped by region.

- Makes a file `SORT_DIR/_stats.h5` containing means and standard deviations of the real-valued features, and counts of the different values for the categorical features.

This will take a while (20 minutes on my fast laptop, 40 minutes on my slow one). Luckily you should only need to do it once per ACS file.


### Featurization

**TODO**


### Analysis

**TODO**
