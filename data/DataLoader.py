from torchtext.data import Field, TabularDataset, Iterator
from transformers import BertTokenizer


class BertDataset(TabularDataset):
    def __init__(
        self,
        PATH,
        tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
        MAX_SEQ_LEN=7,
    ):
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.tokenizer = tokenizer
        self.PAD_INDEX = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.UNK_INDEX = self.tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
        self.LEXICALITY = Field(sequential=False)
        self.TEXT = Field(
            use_vocab=False,
            tokenize=tokenizer.encode,
            lower=False,
            include_lengths=False,
            batch_first=True,
            fix_length=self.MAX_SEQ_LEN,
            pad_token=self.PAD_INDEX,
            unk_token=self.UNK_INDEX,
        )
        self.fields = [
            ("spelling", self.TEXT),
            ("lexicality", self.LEXICALITY),
        ]
        super().__init__(
            self, path=PATH, format="csv", fields=self.fields, skip_header=True
        )


class BertIterator(Iterator):
    def __init__(
        self, dataset, device, batch_size=64, train=True, shuffle=False, sort=False
    ):
        super().__init__(
            self,
            batch_size=batch_size,
            device=device,
            train=train,
            sort=sort,
            shuffle=shuffle,
        )
