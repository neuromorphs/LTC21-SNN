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

def scale_datasets(data):
    '''Scale inputs in a list of datasets to -1, 1'''
    # Find the absolute maximum value for each state 
    pos_bound = max([x['position'].abs().max() for x in data])
    vel_bound = max([x['positionD'].abs().max() for x in data])
    angle_vel_bound = max([x['angleD'].abs().max() for x in data])

    # Scale all datasets to [-1, 1] based on the maximum value found above
    for df in data:
        df['position'] = minmax(df['position'], pos_bound)
        df['positionD'] = minmax(df['positionD'], vel_bound)
        df['angleD'] = minmax(df['angleD'], angle_vel_bound)
    return data