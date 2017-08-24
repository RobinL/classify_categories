import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

def get_data():
    df = pd.read_csv("data_for_model.csv", low_memory=False)

    # Binary classifier - need to detect 'y', which is when the category contains ecstasy pills
    f1 = df["category_concat_cat"].str.lower().str.contains("ecsta")
    f2 = df["category_concat_cat"].str.lower().str.contains("pill")
    
    f3 = f1 & f2
    df.loc[f3, "y"] = 1
    df.loc[~f3, "y"] = 0
    
    # Overrides for listings which are obvious ecstasy pills, but sometimes are not categorised as such
    f1 = df.listing_text.str.lower().str.contains("ecstasy")
    f2 = df.listing_text.str.contains("pill")
    df.loc[f1 & f2, "y"] = 1
    
    f1 = df.listing_text.str.lower().str.contains("mdma")
    df.loc[f1 & f2, "y"] = 1
    
    f1 = df.listing_text.str.lower().str.contains("xtc")
    df.loc[f1 & f2, "y"] = 1
    
    df = df[pd.notnull(df["listing_text"])]

    return df

def get_xy(df):
    y = df["y"]
    x = df[["listing_text", "category_concat_cat"]]
    return (x,y)

df = get_data()
x,y = get_xy(df)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

x_train_all = x_train
x_train = x_train["listing_text"]

x_test_all = x_test
x_test = x_test["listing_text"]