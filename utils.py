import pandas as pd


def transform_data_to_time_series_format(df: pd.DataFrame, instance_id: int):
    df.index = pd.MultiIndex.from_product([[instance_id], range(len(df))], names=["ID", "time"])
    df.columns = ["c1", "c2", "c3"]
    return df
