import xarray as xr
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from typing import Tuple


def get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str)
    args = vars(parser.parse_args())

    return args


def create_results(run_dir: Path) -> Tuple[xr.Dataset, ...]:
    val_path = ([d for d in (run_dir / "test").glob("*")])
    val_results: Dict = pickle.load(
        open(val_path / "validation_results.p", 'rb')
    )
    train_data: Dict = pickle.load(
        open(run_dir / "train_data/train_data.p", 'rb')
    )

    train_ds = create_training_dataset(train_data)
    valid_ds = create_validation_dataset(val_results)

    return train_ds, valid_ds


def create_training_dataset(train_data: Dict) -> xr.Dataset:
    # CREATE TRAINING dataset
    basins = train_data['coords']['basin']['data']
    time = train_data['coords']['date']['data']

    coords = {'station_id': basins, 'time': time}
    dims = ('station_id', 'time')
    data =

    # create the xarray data
    data = {
        variable: (dims, train_data['data_vars'][variable]['data'])
        for variable in [v for v in train_data['data_vars'].keys()]
    }
    train_ds = xr.Dataset(data, coords=coords)

    return train_ds


def create_validation_dataset(val_results: Dict) -> xr.Dataset:
    # create VALIDATION dataset

    station_ids = [stn for stn in val_results.keys()]

    discharge_spec_obs_ALL = []
    discharge_spec_sim_ALL = []

    for stn in station_ids:
        discharge_spec_obs_ALL.append(
            val_results[stn]['xr']['discharge_spec_obs'].values.flatten()
        )
        discharge_spec_sim_ALL.append(
            val_results[stn]['xr']['discharge_spec_sim'].values.flatten()
        )

    times = val_results[stn]['xr']['date'].values
    obs = np.vstack(discharge_spec_obs_ALL)
    sim = np.vstack(discharge_spec_sim_ALL)

    assert obs.shape == sim.shape

    # create xarray object
    coords = {"time": times, "station_id": station_ids}
    data = {
        "obs": (["station_id", "time"], obs),
        "sim": (["station_id", "time"], sim),
    }
    valid_ds = xr.Dataset(data, coords=coords)

    return valid_ds



data_dir = Path("camels_gb_only_global_basin_subset_input_subset_1805_1040")

val_results = pickle.load(
    open(data_dir / "validation/model_epoch030/validation_results.p", 'rb')
)
train_data = pickle.load(
    open(data_dir / "train_data/train_data.p", 'rb')
)


valid_ds.to_netcdf('valid_ds.nc')
train_ds.to_netcdf('train_ds.nc')


valid_ds = xr.open_dataset('valid_ds.nc')
valid_ds.to_dataframe()


if __name__ == "__main__":
    args = get_args()
    run_dir = Path(args["run_dir"])
    assert run_dir.exists()

    train_ds, valid_ds = create_results(run_dir)

    # save to netcdf
    train_ds.to_netcdf('train_ds.nc')
    valid_ds.to_netcdf('valid_ds.nc')
