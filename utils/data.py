import os
import pandas as pd

from utils.scaling import minmax

def load_datasets(data_dir, bound=0.19):
    '''Load a list of datasets from a given data directory'''
    data = []
    for _, _, files in os.walk(data_dir):
        for f in files:
            df = pd.read_csv(data_dir + f, skiprows=28)
            # Filter datasets that might have bounced of the edges of the track
            if df["position"].min() < -bound or df["position"].max() > bound:
                continue
            data.append(df)
    return data

def scale_datasets(data, state_vars):
    '''Scale inputs in a list of datasets to -1, 1'''

    scales = {}
    # Find the absolute maximum value for each state
    for var in state_vars:
        bound = max([x[var].abs().max() for x in data])
        scales.update({var: bound})

    # Scale all datasets to [-1, 1] based on the maximum value found above
    for df in data:
        for var, bound in scales.items():
            df[var] = minmax(df[var], bound)

    # return scaled data and scaling factor for reconstruction
    return data, scales