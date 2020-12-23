import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PATH = Path(__file__).parents[1] / "Data"

df = pd.read_csv(PATH / "blp-items.csv")
df = df[["spelling", "lexicality"]]

train, dev = train_test_split(df, test_size=0.3, random_state=22)
dev, test = train_test_split(dev, test_size=0.33, random_state=22)

test.to_csv(PATH / "test.csv", index=False)
train.to_csv(PATH / "train.csv", index=False)
dev.to_csv(PATH / "dev.csv", index=False)
