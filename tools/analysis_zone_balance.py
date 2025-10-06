import argparse
from pathlib import Path
import pandas as pd


def load_netpos(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {'time', 'zone', 'net_position_mw'}.issubset(df.columns):
        raise ValueError(f"unexpected columns in {path}")
    df['time'] = pd.to_datetime(df['time'])
    grp = df.groupby('zone')['net_position_mw'].agg(['mean', 'sum'])
    grp.columns = [f'{value_name}_mean_mw', f'{value_name}_sum_mwh']
    return grp


def load_loads(load_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(load_dir.glob('load_*_2024.csv')):
        zone = path.stem.replace('load_', '').replace('_2024', '')
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        total_mwh = df['ActualTotalLoad_MW'].sum()
        rows.append((zone, total_mwh))
    return pd.DataFrame(rows, columns=['zone', 'load_sum_mwh']).set_index('zone')


def load_generation(gen_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(gen_dir.glob('actual_gen_*_2024.csv')):
        zone = path.stem.replace('actual_gen_', '').replace('_2024', '')
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        value_cols = [c for c in df.columns if c != 'timestamp']
        total = df[value_cols].fillna(0.0).sum(axis=1).sum()
        rows.append((zone, total))
    return pd.DataFrame(rows, columns=['zone', 'generation_sum_mwh']).set_index('zone')


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarise zonal balances using net positions, load, and generation')
    parser.add_argument('--netpos-scheduled', required=True)
    parser.add_argument('--netpos-physical', required=True)
    parser.add_argument('--load-dir', required=True)
    parser.add_argument('--gen-dir', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()

    sched = load_netpos(Path(args.netpos_scheduled), 'scheduled')
    phys = load_netpos(Path(args.netpos_physical), 'physical')
    loads = load_loads(Path(args.load_dir))
    gens = load_generation(Path(args.gen_dir))

    df = pd.concat([sched, phys, loads, gens], axis=1, sort=True)
    df['netpos_mean_delta_mw'] = df['physical_mean_mw'] - df['scheduled_mean_mw']
    df['netpos_sum_delta_mwh'] = df['physical_sum_mwh'] - df['scheduled_sum_mwh']
    df['generation_minus_load_mwh'] = df['generation_sum_mwh'] - df['load_sum_mwh']

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / 'zone_balance_summary.csv'
    df.reset_index().to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
