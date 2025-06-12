import random
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import List, Sequence, Optional
from gensim.models import KeyedVectors

def random_deletion(tokens: List[str], p: float = 0.1) -> List[str]:
    """
    Drop each token with probability p.
    """
    if len(tokens) == 1:
        return tokens
    kept = [t for t in tokens if random.random() > p]
    return kept or [random.choice(tokens)]


def replace_tokens(
    tokens: List[str],
    n_swaps: int = 1,
    w2v_model: Optional[KeyedVectors] = None
) -> List[List[str]]:
    """
    Generate `n_swaps` copies of `tokens`, each with exactly one
    randomly chosen token swapped for its most similar word.
    
    Returns a list of length `n_swaps`, where each entry is a new token list.
    """
    augmented = []
    print("token_replace!!!")
    for _ in range(n_swaps):
        new_tokens = tokens.copy()
        idx = random.randrange(len(tokens))
        word = tokens[idx]
        # Only attempt replacement if model is provided and word is in vocab
        if w2v_model is not None and word in w2v_model:
            # get the single most similar word
            sim = w2v_model.most_similar(word, topn=1)
            if sim:
                new_tokens[idx] = sim[0][0]
        augmented.append(new_tokens)
    return augmented

def augment_df(
        df: pd.DataFrame,
        augment_threshold: int,
        copies_per_sample: int,
        w2v_model: Optional[KeyedVectors]
    ) -> pd.DataFrame:
        """
        return a new DataFrame where classes with < augment_threshold
        examples get `copies_per_sample` new rows each via replace_tokens.
        """
        counts = df.title.value_counts()
        new_rows = []

        for lbl, cnt in counts.items():
            if cnt < augment_threshold:
                subset = df[df.title == lbl]
                for _, row in subset.iterrows():
                    originals = row.tokens
                    variants = replace_tokens(
                        originals,
                        n_swaps=copies_per_sample,
                        w2v_model=w2v_model
                    )
                    for toks in variants:
                        # copy the row dict and replace tokens
                        row.tokens = toks
                        new_rows.append(row)

        if not new_rows:
            return df.copy()

        aug_df = pd.DataFrame(new_rows)
        # combine original + augmented
        return pd.concat([df, aug_df], ignore_index=True)
