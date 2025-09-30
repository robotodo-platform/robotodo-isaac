
from typing import Sequence
import warnings

import fsspec
import sentencepiece
import jax
# import numpy as np


class PaligemmaTokenizer:
    def __init__(self):
        # TODO better handling
        import platformdirs
        with fsspec.open(
            "filecache::https://storage.googleapis.com/big_vision/paligemma_tokenizer.model", 
            mode="rb", 
            # token="anon",
            cache_storage=platformdirs.user_cache_dir(appname="paligemma_tokenizer.big_vision.todo", appauthor=False),
            same_names=True,
        ) as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    @property
    def vocab_size(self):
        return self._tokenizer.GetPieceSize()
    
    # TODO
    def __call__(self, batch_text: Sequence[str]):
        batch_text = [
            # tokenize "\n" separately as the "start of answer" token
            str.strip(text).replace("_", " ").replace("\n", " ") + "\n"
            for text in batch_text
        ]
        return self._tokenizer.Encode(batch_text, add_bos=True)
