import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Splitter")
parser.add_argument("--destination", "-d", type=str, required=True)
parser.add_argument("--csv", "-c", type=str, required=True)
parser.add_argument("--split", "-s", type=str, default=0.7)
args = parser.parse_args()


DATA_PATH = Path(__file__).parents[1] / "Data"
CREATE_PATH = DATA_PATH / args.destination
if not os.path.exists(CREATE_PATH):
    os.mkdir(CREATE_PATH)

df = pd.read_csv(DATA_PATH / args.csv)
df = df[["spelling", "lexicality"]]

train, dev = train_test_split(df, test_size=1 - args.split, random_state=22)
dev, test = train_test_split(dev, test_size=0.33, random_state=22)

test.to_csv(CREATE_PATH / "test.csv", index=False)
train.to_csv(CREATE_PATH / "train.csv", index=False)
dev.to_csv(CREATE_PATH / "dev.csv", index=False)
