"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from codebase.data.basedatasetbasin import BaseDatasetBasin
from codebase.data.utils import load_camels_attributes, load_discharge, load_forcings

class CamelsGBCSV(BaseDatasetBasin):
    def __init__(self,
                 basin: str,
                 cfg: dict,
                 mode: str,
                 additional_features: List[pd.DataFrame] = [],
                 id_to_int: dict = {},
                 scaler: dict = {}):

        super(CamelsGBCSV, self).__init__(basin=basin,
                                        cfg=cfg,
                                        mode=mode,
                                        additional_features=additional_features,
                                        id_to_int=id_to_int,
                                        scaler=scaler)

        if (isinstance(cfg['dynamic_inputs'], list)) and (len(cfg['dynamic_inputs']) == 1):
            self.forcings = cfg['dynamic_inputs'][0]
        else:
            self.forcings = cfg['dynamic_inputs']

        self.camels_attributes = cfg.get("camels_attributes", [])

        if self.camels_attributes:
            self.attributes = self._load_attributes()

        self.x_d, self.x_s, self.y = self._preprocess_data()

        self.num_samples = self.x_d.shape[0]


    def _get_one_basin_csv_file(self) -> Path:
        # get gauge_id
        gauge_ids = [
            d.name.split("ies_")[-1].split("_")[0]
            for d in (self.data_dir / "Catchment_Timeseries").glob("*.csv")
        ]
        bool_list = [str(id) == str(self.basin) for id in gauge_ids]
        assert (
            sum(bool_list) == 1
        ), f"Only expect to find one gauge id with id : {self.basin}"

        csv_file = np.array([d for d in (self.data_dir / "Catchment_Timeseries").glob("*.csv")])[bool_list][
            0
        ]

        return csv_file

    def _load_data(self) -> pd.DataFrame:
        """Load input and output data from text files."""

        csv_file = self._get_one_basin_csv_file()
        df = pd.read_csv(csv_file)

        # set datatypes
        df["date"] = df["date"].astype(np.dtype("datetime64[ns]"))
        df = df.set_index("date")
        df = df.astype("float")

        # get dynamic_Variables and target_variable
        df = df.loc[:, self.forcings + self.target_variable]

        # replace invalid discharge values by NaNs
        # TODO: does this actually work?
        discharge_cols = [col for col in df.columns if "discharge" in col.lower()]
        for col in discharge_cols:
            df.loc[df[col] < 0, col] = np.nan

        return df

    def _load_attributes(self) -> torch.Tensor:
        df = load_camels_attributes(self.data_dir, [self.basin])

        drop_cols = [c for c in df.columns if c not in self.camels_attributes]

        df = df.drop(drop_cols, axis=1)

        # fix the order of the columns to be alphabetically
        df = df.sort_index(axis=1)

        if not self.is_train:
            # normalize data
            df = (df - self.scaler['camels_attr_mean']) / self.scaler["camels_attr_std"]
        else:
            # we don't have to normalize here, since during training the features are loaded again
            # in the CamelsH5 class and are normalized there based on all training basins
            pass

        # store feature as PyTorch Tensor
        attributes = df.loc[df.index == self.basin].values.flatten()
        return torch.from_numpy(attributes.astype(np.float32))


class CamelsTXT(BaseDatasetBasin):

    def __init__(self,
                 basin: str,
                 cfg: dict,
                 mode: str,
                 additional_features: List[pd.DataFrame] = [],
                 id_to_int: dict = {},
                 scaler: dict = {}):

        super(CamelsTXT, self).__init__(basin=basin,
                                        cfg=cfg,
                                        mode=mode,
                                        additional_features=additional_features,
                                        id_to_int=id_to_int,
                                        scaler=scaler)

        if (isinstance(cfg['forcings'], list)) and (len(cfg['forcings']) == 1):
            self.forcings = cfg['forcings'][0]
        else:
            self.forcings = cfg['forcings']

        self.camels_attributes = cfg.get("camels_attributes", [])

        if self.camels_attributes:
            self.attributes = self._load_attributes()

        self.x_d, self.x_s, self.y = self._preprocess_data()

        self.num_samples = self.x_d.shape[0]

    def _load_data(self):
        """Load input and output data from text files."""
        # get forcings
        if isinstance(self.forcings, list):
            dfs = []
            for forcing in self.forcings:
                df, area = load_forcings(self.data_dir, self.basin, forcing)
                # rename columns
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
                dfs.append(df)
            df = pd.concat(dfs, axis=1)
        else:
            df, area = load_forcings(self.data_dir, self.basin, self.forcings)

        # add discharge
        df['QObs(mm/d)'] = load_discharge(self.data_dir, self.basin, area)

        # replace invalid discharge values by NaNs
        qobs_cols = [col for col in df.columns if "qobs" in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        return df

    def _load_attributes(self) -> torch.Tensor:
        df = load_camels_attributes(self.data_dir, [self.basin])

        drop_cols = [c for c in df.columns if c not in self.camels_attributes]

        df = df.drop(drop_cols, axis=1)

        # fix the order of the columns to be alphabetically
        df = df.sort_index(axis=1)

        if not self.is_train:
            # normalize data
            df = (df - self.scaler['camels_attr_mean']) / self.scaler["camels_attr_std"]
        else:
            # we don't have to normalize here, since during training the features are loaded again
            # in the CamelsH5 class and are normalized there based on all training basins
            pass

        # store feature as PyTorch Tensor
        attributes = df.loc[df.index == self.basin].values.flatten()
        return torch.from_numpy(attributes.astype(np.float32))
