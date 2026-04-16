import json
import time
import pandas as pd
import seaborn as sns
from scipy.signal import butter, filtfilt

from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.optimize import fmin
import copy
import numpy as np
import datetime
import random
import string
from copy import deepcopy

def normalize_df_period(periodLengthInMS, samplingRateUS, df, interpolationKind='cubic'):
    """
    Normates the period of a given dataset DataFrame.
    
    The input DataFrame should have columns: timestamp, x, y, z.
    The 'timestamp' column represents the time difference in microseconds from the previous timestamp.
    This function converts the differences into absolute time values, then interpolates the dataset
    so that the period lasts for periodLengthInMS (in milliseconds) with data points every samplingRateUS (in µs).
    
    Parameters:
        periodLengthInMS (number): Target period length in milliseconds.
        samplingRateUS (number): Sampling rate in microseconds for the interpolated data.
        df (DataFrame): Input DataFrame with columns ['timestamp', 'x', 'y', 'z'].
        interpolationKind (str): Interpolation method (default 'cubic'; other options include 'linear', 
                                 'nearest', 'zero', 'slinear', 'quadratic', etc.).
    
    Returns:
        DataFrame: An interpolated DataFrame with columns ['timestamp', 'x', 'y', 'z'].
                   The 'timestamp' column in the output starts at 0 and increments by samplingRateUS up to the target period.
    """
    # Convert target period from milliseconds to microseconds.
    target_period_us = periodLengthInMS * 1000

    # Create a copy of the dataframe to avoid modifying the original.
    df = df.copy()

    # Convert the timestamp differences to absolute timestamps.
    df['abs_timestamp'] = df['timestamp'].cumsum()

    # Determine the original period length from the cumulative timestamp.
    original_period_us = df['abs_timestamp'].iloc[-1]

    # Create interpolation functions for x, y, and z based on the absolute timestamps.
    f_x = interp1d(df['abs_timestamp'].values, df['a_x'].values, kind=interpolationKind,
                   bounds_error=False, fill_value=(df['a_x'].iloc[0], df['a_x'].iloc[0]))
    f_y = interp1d(df['abs_timestamp'].values, df['a_y'].values, kind=interpolationKind,
                   bounds_error=False, fill_value=(df['a_y'].iloc[0], df['a_y'].iloc[0]))
    f_z = interp1d(df['abs_timestamp'].values, df['a_z'].values, kind=interpolationKind,
                   bounds_error=False, fill_value=(df['a_z'].iloc[0], df['a_z'].iloc[0]))

    # Prepare list to hold new (normalized) data rows.
    interpolated_data = []
    sample_moment = 0  # New timestamp in output (in µs)

    if original_period_us <= target_period_us:
        # If the original period is shorter than (or equal to) the target period,
        # use the entire range from 0 to target_period_us.
        while sample_moment <= target_period_us:
            x_val = float(f_x(sample_moment))
            y_val = float(f_y(sample_moment))
            z_val = float(f_z(sample_moment))
            interpolated_data.append([int(sample_moment), x_val, y_val, z_val])
            sample_moment += samplingRateUS
    else:
        # If the original period is longer than the target, center the extracted data.
        # Calculate how much time to trim from the start (and end).
        time_to_cutoff = round((original_period_us - target_period_us) / 2)
        pseudo_sample_moment = time_to_cutoff
        while sample_moment <= target_period_us:
            x_val = float(f_x(pseudo_sample_moment))
            y_val = float(f_y(pseudo_sample_moment))
            z_val = float(f_z(pseudo_sample_moment))
            interpolated_data.append([int(sample_moment), x_val, y_val, z_val])
            pseudo_sample_moment += samplingRateUS
            sample_moment += samplingRateUS

    # Create a new DataFrame for the interpolated data.
    interpolated_df = pd.DataFrame(interpolated_data, columns=['timestamp', 'a_x', 'a_y', 'a_z'])
    return interpolated_df


def butter_lowpass_filter_df(df, fs=70.0, cutoff=30.0, order=4):

    # Copy to avoid mutating original
    df_filtered = df.copy()
    
    # Design a Butterworth filter
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Filter each axis using zero-phase filtering
    for axis in ['a_x', 'a_y', 'a_z']:
        df_filtered[axis] = filtfilt(b, a, df_filtered[axis].values)
    
    return df_filtered
