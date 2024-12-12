from pathlib import Path
from typing import List
import torch
from fontTools.ttLib import TTFont

from text2lis.model.colator import zero_pad_collator

MAX_TEXT_LEN = 100

class EnglishTokenizer:

    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.unk_token_id = 2  # Token for unknown characters

        # Use a standard font that supports English characters (Arial)
        self.font_path = r"arial.ttf"
        
        with TTFont(self.font_path) as font:
            tokens = [chr(key) for key in font["cmap"].getBestCmap().keys()]

        # Create a mapping from integer ids to characters and vice versa
        self.i2s = {(i + 3): c for i, c in enumerate(tokens)}
        self.s2i = {c: i for i, c in self.i2s.items()}
        self.s2i['[UNK]'] = self.unk_token_id  # Add token for unknown characters
        self.i2s[self.unk_token_id] = '[UNK]'

    def __len__(self):
        return len(self.i2s) + 2

    def tokenize(self, text: str):
        return [self.bos_token_id] + [self.s2i.get(c, self.unk_token_id) for c in text]

    def __call__(self, texts: List[str], device=None):
        all_tokens = [self.tokenize(text) for text in texts]

        tokens_batch = zero_pad_collator([{
            "tokens_ids": torch.tensor(tokens, dtype=torch.long, device=device),
            "attention_mask": torch.ones(len(tokens), dtype=torch.bool, device=device),
            "positions": torch.arange(0, len(tokens), dtype=torch.long, device=device)
        } for tokens in all_tokens])
        # In transformers, 1 is mask, not 0
        tokens_batch["attention_mask"] = torch.logical_not(tokens_batch["attention_mask"])

        return tokens_batch

if __name__ == "__main__":
    tokenizer = EnglishTokenizer()
    english_texts = [
        "ciao come stai?",
        "ciao"

    ]
    print(english_texts)
    print(tokenizer(english_texts))
