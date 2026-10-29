import pandas as pd


def lower(df):
    return df.apply(lambda c: c.str.lower() if c.dtype == object else c)


subtype = pd.read_csv("../data/embryos/final_subtype_v2.csv", index_col=0)
lower(subtype).to_csv("../data/embryos/final_subtype_v2_lower.csv")

info = pd.read_csv("../data/embryos/final_type_info_v2.csv")
lower(info).to_csv("../data/embryos/final_type_info_v2_lower.csv", index=False)
