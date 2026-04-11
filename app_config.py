# app_config.py

import os
import json
import pandas as pd
import xarray as xr
import streamlit as st
from pathlib import Path

from graphcast_structure_1 import (
    get_available_months_gc_1,
    get_available_times_gc_1,
    load_latent_gc_1,
)

CONFIG_PATH = Path("paths.json")

def load_paths():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Missing paths.example.json. Copy paths.example.json to paths.json and edit it."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

PATHS = load_paths()

APP_DEFAULTS = {
    "latent_dim": 512,
    "batch_num": 0,
    "processor_step_max": 16,
    "figsize": (11, 7),
    "figsize_hist": (7, 5),
    "dpi": 100,
    "col_1_width": 3,
    "col_2_width": 2,
    "default_model": "graphcast_small",
    "default_latent_label": "flt",   # or "fit"
    "default_use_translator": True,
    "default_n_top": 9,
}

MODELS = {
    "graphcast_small": {
        "label": "GraphCast small",
        "latent": {
            "dir": PATHS["graphcast_small"]["latent_dir"],
            "translator_dir": PATHS["graphcast_small"]["translator_dir"],
            "structure": "graphcast_structure_1",
            "label": "flt",   # hardcoded here, not user-selected
            "use_translator_default": True,
            "processor_step_max": 16,
        },
        "era5": {
            "filename": "Graphcast_small_processed_input_{ym_str}.nc",
            "basepath": PATHS["graphcast_small"]["era5_basepath"]},
        "graph_coords_filepath": PATHS["graphcast_small"]["graph_coords_filepath"],
        "translator_template": "translator_matrix_{step}_gnn.npz",
        "timestep_start_hour": 12,
        "timestep_hours": 6,
    },
}

# match models with functions to use for loading data 

def get_available_months(model_cfg):
    structure = model_cfg["latent"]["structure"]

    if structure == "graphcast_structure_1":
        return get_available_months_gc_1(model_cfg)
#    elif structure == "other_structure_a":
#        return get_available_months_other_a(model_cfg)
    else:
        raise ValueError(f"Unknown latent structure: {structure}")

def get_available_times(year, month, model_cfg):
    structure = model_cfg["latent"]["structure"]

    if structure == "graphcast_structure_1":
        return get_available_times_gc_1(year, month, model_cfg)
    # elif structure == "other_structure_a":
    #     return get_available_times_other_a(year, month, model_cfg)
    else:
        raise ValueError(f"Unknown latent structure: {structure}")

@st.cache_data(show_spinner=False)
def load_latent(flt_time, model_cfg, batch_num, use_translator=False):
    structure = model_cfg["latent"]["structure"]
    if structure == "graphcast_structure_1":
        return load_latent_gc_1(flt_time, model_cfg, batch_num, use_translator)
    # elif structure == "other_structure_a":
    #     return load_latent_other_a(year, month, day, hour, model_cfg, use_translator)
    else:
        raise ValueError(f"Unknown latent structure: {structure}")


@st.cache_data(show_spinner=False)
def load_era5_fit_and_flt(flt_time, model_cfg):
    """
    Load ERA5 for a month and return fit and flt slices using actual timestamps.

    flt_time: pandas.Timestamp, UTC
    Returns:
        ds_pair with 2 times: [fit, flt]
    """
    flt_time = pd.Timestamp(flt_time)
    if flt_time.tz is not None:
        flt_time = flt_time.tz_convert("UTC").tz_localize(None)

    fit_time = flt_time - pd.Timedelta(hours=6)

    ym_str = f"{fit_time.year}{fit_time.month:02d}"
    era5_filename = model_cfg["era5"]["filename"].format(ym_str=ym_str)
    era5_basepath = model_cfg["era5"]["basepath"]
    era5_filepath = os.path.join(era5_basepath, era5_filename)

    ds = xr.open_dataset(era5_filepath)

    # Make sure ERA5 time is comparable
    ds = ds.assign_coords(time=pd.to_datetime(ds.time.values))

    ds_pair = ds.sel(time=[fit_time, flt_time])

    if ds_pair.sizes["time"] != 2:
        raise ValueError(f"Could not find both fit={fit_time} and flt={flt_time} in ERA5 file.")

    return ds_pair


