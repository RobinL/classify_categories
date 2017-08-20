import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

def get_data():
    df = pd.read_csv("data_for_model.csv", low_memory=False)

    # Binary classifier - need to detect
    df_dedup = df.drop_duplicates("category_concat_cat")
    f1 = df_dedup["category_concat_cat"].str.lower().str.contains("ecsta")
    f2 = df_dedup["category_concat_cat"].str.lower().str.contains("pill")
    ccc = list(df_dedup[f1&f2]["category_concat_cat"])
    df.loc[df.category_concat_cat.isin(ccc), "y"] = 1
    df.loc[~(df.category_concat_cat.isin(ccc)), "y"] = 0

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