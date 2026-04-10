# Graphcast structure 1 functions #
import streamlit as st
import re
import os
import pandas as pd
import numpy as np
from utils import apply_translator

def get_available_months_gc_1(model_cfg):
    latent_dir = model_cfg["latent"]["dir"]
    latent_label = model_cfg["latent"]["label"]   # e.g. "flt"

    pattern = re.compile(rf"latent_mesh_step_(\d+)_(\d{{4}})_(\d{{2}})\.npz")

    months = []
    for f in os.listdir(latent_dir):
        m = pattern.match(f)
        if m:
            _, year, month = m.groups()
            months.append((int(year), int(month)))

    return sorted(set(months))

def get_available_times_gc_1(year, month, model_cfg):
    if year is None or month is None:
        raise ValueError(
            f"Invalid year/month passed to get_available_times_gc_1: "
            f"year={year}, month={month}"
        )

    start_time = pd.Timestamp(year, month, 1, 12, 0, tz="UTC")

    if month == 12:
        end_time = pd.Timestamp(year + 1, 1, 1, 0, 0, tz="UTC")
    else:
        end_time = pd.Timestamp(year, month + 1, 1, 0, 0, tz="UTC")

    return pd.date_range(start=start_time, end=end_time, freq="6h", inclusive="left")

def load_latent_gc_1(flt_time, model_cfg, batch_num=0, use_translator=False):
    latent_dir = model_cfg["latent"]["dir"]
    translator_dir = model_cfg["latent"]["translator_dir"]

    year = flt_time.year
    month = flt_time.month

    # --- compute timestep index ---
    start_time = pd.Timestamp(year, month, 1, 12, 0, tz="UTC")
    time_index = int((flt_time - start_time) / pd.Timedelta(hours=6))

    pattern = re.compile(rf"latent_mesh_step_(\d+)_(\d{{4}})_(\d{{2}})\.npz")

    parsed = []
    for f in os.listdir(latent_dir):
        m = pattern.match(f)
        if m:
            step, y, mo = m.groups()
            if int(y) == year and int(mo) == month:
                parsed.append({"filename": f, "step": int(step)})

    parsed = sorted(parsed, key=lambda x: x["step"])
    if not parsed:
        raise FileNotFoundError(f"No latent files found for {year}-{month:02d}")

    latent_list = []
    steps = []

    for p in parsed:
        filename = os.path.join(latent_dir, p["filename"])
        arr = np.load(filename)
        key = arr.files[0]
        full_latent = arr[key]   # (timesteps, nodes, batch, latent_dim)

        latent_t = full_latent[time_index, :, batch_num, :]

        if use_translator and p["step"] != 16:
            translator_file = os.path.join(
                translator_dir,
                f"translator_matrix_{p['step']}_gnn.npz"
            )
            latent_t = apply_translator(latent_t, translator_file)

        latent_list.append(latent_t)
        steps.append(p["step"])

    latent_out = np.stack(latent_list, axis=0)  # (processor steps, nodes, latent dimension)
    return latent_out, steps, time_index
