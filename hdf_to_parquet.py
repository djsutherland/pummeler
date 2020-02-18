#!/usr/bin/env python

from pathlib import Path
import shutil

import pandas as pd
from tqdm import tqdm

from pummeler.stats import load_stats


def convert(in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    stats = load_stats(in_dir / 'stats.h5')
    info = stats['version_info']

    sources = [p for p in in_dir.glob("feats_*.h5")
               if not (out_dir / (p.stem + '.parquet')).exists()]

    for fn in tqdm(sources):
        df = pd.read_hdf(fn)

        w = info['alloc_flags'] + info['weight_cols']
        df[w] = df[w].apply(pd.to_numeric, downcast='unsigned')

        for k, vc in stats['value_counts'].items():
            df[k] = pd.Categorical(df[k], categories=vc.index)
        df['RT'] = pd.Categorical(df['RT'])
        
        df.to_parquet(out_dir / (fn.stem + '.parquet'))

    stats_out = out_dir / 'stats.h5'
    if stats_out.exists():
        print(f"{stats_out} already exists, not overriding")
    else:
        shutil.copy(in_dir / 'stats.h5', stats_out)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=Path)
    parser.add_argument('out_dir', nargs='?', type=Path)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.in_dir.with_name(args.in_dir.name + '_pq')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {args.in_dir} to {args.out_dir}...")
    convert(args.in_dir, args.out_dir)

if __name__ == '__main__':
    main()
