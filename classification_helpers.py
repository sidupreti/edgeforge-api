import pandas as pd
import numpy as np

def compute_minimal_features(df, group_col="punch_id", sort_col="timestamp", value_cols=["a_x", "a_y", "a_z"]):
    features_list = []
    
    # Group by the given identifier (e.g. punch_id)
    grouped = df.groupby(group_col)
    
    for group_id, group_df in grouped:
        # sort the group by the timestamp column
        group_df = group_df.sort_values(by=sort_col)
        feats = {}
        for col in value_cols:
            series = group_df[col].to_numpy()
            feats[f"{col}__sum_values"] = np.float64(np.sum(series))
            feats[f"{col}__median"] = np.float64(np.median(series))
            feats[f"{col}__mean"] = np.float64(np.mean(series))
            feats[f"{col}__length"] = np.float64(len(series))
            feats[f"{col}__standard_deviation"] = np.float64(np.std(series, ddof=0))
            feats[f"{col}__variance"] = np.float64(np.var(series, ddof=0))
            feats[f"{col}__root_mean_square"] = np.float64(np.sqrt(np.mean(series**2)))
            feats[f"{col}__maximum"] = np.float64(np.max(series))
            feats[f"{col}__absolute_maximum"] = np.float64(np.max(np.abs(series)))
            feats[f"{col}__minimum"] = np.float64(np.min(series))
        # Optionally include the group id; here we leave it as the index.
        feats[group_col] = group_id
        features_list.append(feats)
    
    # Create the resulting DataFrame and set the group_col as index.
    features_df = pd.DataFrame(features_list).set_index(group_col)
    
    # Ensure the feature columns are float64 (group id remains as index)
    features_df = features_df.astype('float64')
    
    return features_df
