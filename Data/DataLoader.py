from torchtext import data

TEXT = data.Field()
LEXICALITY = data.LabelField()

fields = [
    ("spelling", TEXT),
    ("lexicality", LEXICALITY),
]

# load the dataset in json format
train_ds, valid_ds, test_ds = data.TabularDataset.splits(
    path="./",
    train="train.tsv",
    validation="valid.tsv",
    test="test.tsv",
    format="tsv",
    fields=fields,
    skip_header=True,
)
