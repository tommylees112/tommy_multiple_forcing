import xarray as xr
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from typing import Tuple, Dict


def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str)
    args = vars(parser.parse_args())

    return args


def get_validation_data(run_dir: Path) -> xr.Dataset:
    # open validation Dict
    val_path = [d for d in (run_dir / "test").glob("*")]
    val_path = max(val_path)
    val_results: Dict = pickle.load(open(val_path / "test_results.p", "rb"))

    valid_ds = create_validation_dataset(val_results)
    return valid_ds


def get_train_data(run_dir: Path) -> Tuple[xr.Dataset, ...]:
    # open train Dict
    train_path = [d for d in (run_dir / "train_data").glob("*.p")][0]
    train_data: Dict = pickle.load(open(train_path, "rb"))
    train_ds = create_training_dataset(train_data)

    return train_ds


def create_training_dataset(train_data: Dict) -> xr.Dataset:
    # CREATE TRAINING dataset
    basins = train_data["coords"]["basin"]["data"]
    time = train_data["coords"]["date"]["data"]

    coords = {"station_id": basins, "time": time}
    dims = ("station_id", "time")

    # create the xarray data
    data = {
        variable: (dims, train_data["data_vars"][variable]["data"])
        for variable in [v for v in train_data["data_vars"].keys()]
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
            val_results[stn]["xr"]["discharge_spec_obs"].values.flatten()
        )
        discharge_spec_sim_ALL.append(
            val_results[stn]["xr"]["discharge_spec_sim"].values.flatten()
        )

    times = val_results[stn]["xr"]["date"].values
    obs = np.vstack(discharge_spec_obs_ALL)
    sim = np.vstack(discharge_spec_sim_ALL)

    assert obs.shape == sim.shape

    # create xarray object
    coords = {"time": times, "station_id": station_ids}
    data = {"obs": (["station_id", "time"], obs), "sim": (["station_id", "time"], sim)}
    valid_ds = xr.Dataset(data, coords=coords)

    return valid_ds


if __name__ == "__main__":
    args = get_args()
    run_dir = Path(args["run_dir"])
    assert run_dir.exists()

    valid_ds = get_validation_data(run_dir)

    # save to netcdf
    # train_ds.to_netcdf('train_ds.nc')
    # valid_ds.to_netcdf()
    valid_ds.to_netcdf(run_dir / "valid_ds.nc")
    valid_ds.to_dataframe().to_csv(run_dir / f"results_{run_dir.name}.csv")

    print(f"Results written to {run_dir}")
