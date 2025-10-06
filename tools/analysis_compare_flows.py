import argparse
from pathlib import Path
import pandas as pd


def load_pairwise(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'time' not in df.columns:
        raise ValueError(f"missing 'time' column in {path}")
    df['time'] = pd.to_datetime(df['time'])
    value_cols = [c for c in df.columns if c.startswith('flow_')]
    wide = df.set_index('time')[value_cols]
    renamed = wide.rename(columns=lambda c: c.replace('flow_', ''))
    long = renamed.stack(dropna=False).reset_index()
    long.columns = ['time', 'corridor', value_name]
    return long


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scheduled vs physical ENTSO-E pairwise flows")
    parser.add_argument('--scheduled', required=True, help='CSV with pairwise scheduled flows (A09)')
    parser.add_argument('--physical', required=True, help='CSV with pairwise physical flows (A11)')
    parser.add_argument('--outdir', required=True, help='Directory to write comparison outputs')
    args = parser.parse_args()

    scheduled_path = Path(args.scheduled)
    physical_path = Path(args.physical)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scheduled = load_pairwise(scheduled_path, 'scheduled_mw')
    physical = load_pairwise(physical_path, 'physical_mw')

    merged = pd.merge(scheduled, physical, on=['time', 'corridor'], how='outer')
    merged.sort_values(['corridor', 'time'], inplace=True)
    merged['delta_mw'] = merged['physical_mw'] - merged['scheduled_mw']

    out_file = outdir / 'flow_delta_timeseries.csv'
    merged.to_csv(out_file, index=False)

    summary = merged.groupby('corridor').agg(
        hours=('time', 'count'),
        scheduled_mean=('scheduled_mw', 'mean'),
        physical_mean=('physical_mw', 'mean'),
        delta_mean=('delta_mw', 'mean'),
        delta_abs_mean=('delta_mw', lambda s: s.abs().mean()),
        delta_abs_p90=('delta_mw', lambda s: s.abs().quantile(0.90)),
    ).reset_index()
    summary.sort_values('delta_abs_mean', ascending=False, inplace=True)

    summary_file = outdir / 'flow_delta_summary.csv'
    summary.to_csv(summary_file, index=False)


if __name__ == '__main__':
    main()
