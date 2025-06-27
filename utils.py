import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    return df