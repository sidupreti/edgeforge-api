import pandas as pd
import copy
import numpy as np
import json
from scipy.signal import find_peaks
import itertools
import numpy as np
import pandas as pd

def analyze_sample_rate(df):

    # Make a copy to avoid modifying the original DataFrame.
    df_copy = df.copy()
    
    # Although the differences are in microseconds, we still sort in case order matters.
    df_copy = df_copy.sort_values(by="timestamp")
    
    # Compute the total period length by summing all timestamp differences.
    # (This gives the total elapsed time in µs between the first and last sample.)
    period_length_us = df_copy["timestamp"].sum()
    
    # Convert the period length to seconds.
    period_length_s = period_length_us / 1e6
    
    # The number of intervals between samples is one less than the number of samples.
    num_samples = len(df_copy)
    num_intervals = num_samples - 1
    
    # Compute the sample rate: intervals per second.
    sample_rate_hz = num_intervals / period_length_s if period_length_s > 0 else float('nan')
    
    return sample_rate_hz


def convert_txt_to_dataframe(file_path):
    # Read the text file into a DataFrame with columns: timestamp, x, y, z
    df = pd.read_csv(file_path, names=['timestamp', 'a_x', 'a_y', 'a_z'], header=None)
    
    # Extract the accelerometer data (x, y, z) into a NumPy array.
    acc_data = df[['a_x', 'a_y', 'a_z']].values
    
    # Define the transformation from your orientation matching results:
    permutation = (2, 0, 1)
    sign_flips = (-1, -1, 1)
    
    # Apply the transformation on the accelerometer data.
    aligned_acc = np.column_stack([
        sign_flips[i] * acc_data[:, permutation[i]] for i in range(3)
    ])
    
    # Replace the original accelerometer columns with the re-oriented data.
    df[['a_x', 'a_y', 'a_z']] = aligned_acc

    df["timestamp"] = df["timestamp"].diff().fillna(0).astype(int)
    
    return df


def compute_metrics(df, prominence=1.0, distance=2):

    # Copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Compute the resultant acceleration
    df['resultant'] = np.sqrt(df['a_x']**2 + df['a_y']**2 + df['a_z']**2)
    
    # Find peaks with the specified prominence and minimum distance
    peaks, _ = find_peaks(df['resultant'], prominence=prominence, distance=distance)
    
    # If a prominent peak exists, trim the DataFrame up to a few samples past the first peak
    if len(peaks) > 0:
        first_peak_idx = peaks[0]
        df_trimmed = df.iloc[: first_peak_idx + 5]
    else:
        # No peak found that meets the criteria; use the entire DataFrame
        df_trimmed = df
    
    # Compute maximum acceleration from the resultant
    max_accel = df_trimmed['resultant'].max()
    
    # Compute maximum g-force
    max_g = max_accel / 9.81  # 1g ~ 9.81 m/s^2
    
    # Numerically integrate acceleration to get velocity.
    # Initialize velocities for each axis
    vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
    velocity_magnitudes = []
    
    # Iterate over each sample to update velocity using dt computed from timestamps (in microseconds)
    for i in range(len(df_trimmed) - 1):
        dt = (df_trimmed['timestamp'].iloc[i+1] - df_trimmed['timestamp'].iloc[i]) / 1e6
        # Current acceleration values
        ax = df_trimmed['a_x'].iloc[i]
        ay = df_trimmed['a_y'].iloc[i]
        az = df_trimmed['a_z'].iloc[i]
        
        # Update velocities
        vel_x += ax * dt
        vel_y += ay * dt
        vel_z += az * dt
        
        # Calculate and store the velocity magnitude
        velocity_magnitudes.append(np.sqrt(vel_x**2 + vel_y**2 + vel_z**2))
    
    # Maximum velocity encountered; if no integration steps were performed, default to 0.0
    max_velocity = max(velocity_magnitudes) if velocity_magnitudes else 0.0
    
    # Return the metrics as a dictionary
    return max_accel, max_g, max_velocity


def print_display_metrics(target_name, classified_label, max_a, max_g, max_v):

    if target_name == "TELEGRAPH_R":

        if classified_label == [3]:
            display_label = "Cross"
        elif classified_label == [2]:
            display_label = "Right-Hook"
        else:
            display_label = "Right-Uppercut"

    elif target_name == "TELEGRAPH_L":
        if classified_label == [3]:
            display_label = "Jab"
        elif classified_label == [2]:
            display_label = "Left-Hook"
        else:
            display_label = "Left-Uppercut"

    print(display_label)
    print(f"Max Acceleration:   {max_a:.2f} m/s^2")
    print(f"Max G-Force:        {max_g:.2f} G")
    print(f"Max Velocity:       {max_v:.2f} m/s")

    return None