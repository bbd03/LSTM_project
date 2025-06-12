import pickle
import itertools
from collections import Counter
from typing import List, Sequence, Tuple, Dict
from collections.abc import Sequence as ABCSequence  

import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors

SPECIAL_TOKENS: List[str] = ["<pad>", "<unk>"]
EMB_STD: float = 0.8

class Vocabulary:
    """
    Bidirectional token/index mapping + pretrained embedding matrix.
    """
    def __init__(self, word2idx: Dict[str, int], embed_mat: np.ndarray, origin_len: int):
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}
        self.embed_mat = embed_mat  # shape (V, D)
        self._origin_len = origin_len

    def __len__(self) -> int:
        return len(self.word2idx)

    def __call__(self, token: str) -> int:
        return self.word2idx.get(token, self.word2idx["<unk>"])
    
    def encode(self, tokens: Sequence[str]) -> List[int]:
        """List[str] → List[int]"""
        if not isinstance(tokens, List):
            raise TypeError(f"expected List[str], got {type(tokens)}")

        return [self(t) for t in tokens]

    def decode(self, indices: Sequence[int]) -> List[str]:
        """List[int] → List[str]"""
        if not isinstance(indices, List):
            raise TypeError(f"expected List[int], got {type(indices)}")

        return [self.idx2word.get(i, "<unk>") for i in indices]

    # some diagnostics

    def oov_rate(self, tokens: Sequence[str]) -> float:
        """Return fraction of *tokens* that map to <unk>.
        """

        if not isinstance(tokens, ABCSequence):
            raise TypeError("oov_rate expects a sequence, got " + type(tokens).__name__)
        if tokens and not isinstance(tokens[0], str):
            raise TypeError("oov_rate expects a sequence of str, found " + type(tokens[0]).__name__)

        if len(tokens) == 0:
            return 0.0

        unk_id = self.word2idx["<unk>"]
        oov = sum(1 for t in tokens if self(t) == unk_id)
        return oov / len(tokens)


    def coverage_report(self, kv: KeyedVectors) -> Tuple[int, int, float]:
        """coverage of this vocab relative to a KeyedVectors object."""
        covered = sum(1 for w in self.word2idx if w in kv)
        total = self._origin_len
        return covered, total, covered / total

    def save(self, save_dir: str | Path) -> None:
        """Write word2idx.pkl + embed_mat.npy to *save_dir*."""
        try:
            save_dir = Path(save_dir)
        except:
            raise TypeError(f"expected str or Path, got {type(save_dir)}")

        pkl_path = save_dir / "word2idx.pkl"

        with pkl_path.open("wb") as f:
            pickle.dump(self.word2idx, f, pickle.HIGHEST_PROTOCOL)

        np.save(save_dir / "embed_mat.npy", self.embed_mat)

    @staticmethod
    def load(save_dir: str | Path) -> "Vocabulary":

        save_path = Path(save_dir)
        word2idx = pickle.loads((save_path / "word2idx.pkl").read_bytes())
        embed_mat = np.load(save_path / "embed_mat.npy")
        return Vocabulary(word2idx, embed_mat)


    @staticmethod
    def build_from_dataframe(
        df_tokens,              # pandas Series of List[str]
        kv: KeyedVectors,       # pre‑loaded Word2Vec model
        min_freq: int = 2,
        max_vocab: int | None = 50_000,
        save_dir: str | None = None,
    ) -> "Vocabulary":
        """Create a Vocabulary + embedding matrix from a tokenised corpus.

        Parameters
        ----------
        df_tokens : pandas Series where each row is List[str]
        kv        : gensim KeyedVectors already in memory
        min_freq  : discard tokens that appear < min_freq times
        max_vocab : keep at most this many tokens (incl. specials)
        save_dir  : if given, save word2idx.pkl & embed_mat.npy there
        """ 

        counter = Counter(itertools.chain.from_iterable(df_tokens))
        tokens = [w for w, c in counter.most_common() if c >= min_freq]

        origin_len = len(tokens)
        tokens = [w for w in tokens if w in kv]

        if max_vocab is not None:
            tokens = tokens[: len(tokens) if max_vocab > len(tokens) else max_vocab]

        word2idx: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for w in tokens:
            word2idx[w] = len(word2idx)

        emb_dim = kv.vector_size
        vocab_size = len(word2idx)
        embed_mat = np.random.normal(scale=EMB_STD, size=(vocab_size, emb_dim)).astype(np.float32)
        for w, i in word2idx.items():
            if w in kv:
                embed_mat[i] = kv[w]

        vocab = Vocabulary(word2idx, embed_mat, origin_len=origin_len)

        if save_dir is not None:
            vocab.save(save_dir)

        return vocab


    



    


    


