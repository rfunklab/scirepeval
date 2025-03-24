import pandas as pd
from expansion.defaults import default_config

def train_test_split(df, pct, random_state=default_config['seed']):
    train = df.sample(frac=pct/100, random_state=random_state)
    test = df.drop(train.index)
    return train, test

def train_test_split_by_index(df, pct, random_state=default_config['seed']):
    unique_indices = df.index.unique()
    train_indices = unique_indices.to_series().sample(frac=pct/100, random_state=random_state)
    train = df[df.index.isin(train_indices)]
    test = df[~df.index.isin(train_indices)]
    return train, test