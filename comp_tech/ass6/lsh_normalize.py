#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
import pandas as pd
import csv
import argparse
import time
from typing import Dict, Tuple

def load_glove(filename: str)->Tuple[Dict[str,int],Dict[int,str],
                                    npt.NDArray[np.float64]]:
    """
    Loads the glove dataset. Returns three things:
    A dictionary that contains a map from words to rows in the dataset.
    A reverse dictionary that maps rows to words.
    The embeddings dataset as a NumPy array.
    """
    df = pd.read_table(filename, sep=' ', index_col=0, header=None,
                           quoting=csv.QUOTE_NONE)
    word_to_idx: Dict[str,int] = dict()
    idx_to_word: Dict[int,str] = dict()
    for (i,word) in enumerate(df.index):
        word_to_idx[word] = i
        idx_to_word[i] = word
    return (word_to_idx, idx_to_word, df.to_numpy())

def normalize(X: npt.NDArray[np.float64])->npt.NDArray[np.float64]:
    # Compute the L2 norms of each row
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    
    # Divide each row by its corresponding norm
    normalized_matrix = X / norms
    
    return normalized_matrix
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Glove dataset filename',
                            type=str)
    args = parser.parse_args()
    (word_to_idx, idx_to_word, X) = load_glove(args.dataset)

    start = time.time()

    X = normalize(X)

    end = time.time()

    print(end - start)
