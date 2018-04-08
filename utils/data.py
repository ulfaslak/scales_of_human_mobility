from sensible_raw.loaders import loader
import pandas as pd

def get_datatype_period(datatype, period, sort=False):
    """Get data for a datatype and list of periods."""
    df = pd.DataFrame()
    for month in period:
        df = pd.concat([df, loader.load_data(datatype, month, as_dataframe=True)])

    df.loc[:, 'timestamp'] = df.loc[:, 'timestamp'] / 1000
    if sort:
        df = df.sort_values(["timestamp", "user"])
        df.index = list(range(df.shape[0]))
        
    return df