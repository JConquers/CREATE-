import pandas as pd
url = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"
print("Loading...")
df = pd.read_json(url, compression="gzip", lines=True, nrows=5)
print(df.columns)
print(df.head())
