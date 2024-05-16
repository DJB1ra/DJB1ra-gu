#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
import pandas as pd
import csv
import argparse
import time
from operator import itemgetter
from typing import Dict, Tuple, List, Optional, Set

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
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    normalized_matrix = X / norms
    
    return normalized_matrix

def construct_queries(queries_fn: str, word_to_idx: Dict[str,int],
                          X: npt.NDArray[np.float64]) -> \
                          Tuple[npt.NDArray[np.float64],List[str]]:
    """
    Reads queries (one string per line) and returns:
    - The query vectors as a matrix Q (one query per row)
    - Query labels as a list of strings
    """
    with open(queries_fn, 'r') as f:
        queries = f.read().splitlines()
    Q = np.zeros((len(queries), X.shape[1]))
    for i in range(len(queries)):
        Q[i,:] = X[word_to_idx[queries[i]],:]
    return (Q,queries)

class RandomHyperplanes:
    """
    This class mimics the interface of sklearn:
    - the constructor sets the number of hyperplanes
    - the random hyperplanes are drawn when fit() is called 
      (input dimension is set)
    - transform actually transforms the vectors
    - fit_transform does fit first, followed by transform
    """
    def __init__(self, d: int, seed: Optional[int] = None)->None:
        """
        Sets the number of hyperplanes (d) and the optional random number seed
        """
        self._d = d
        self._seed = seed
        self._hyperplanes = None

    def fit(self, X: npt.NDArray[np.float64])->None:
        """
        Draws _d random hyperplanes, that is, by drawing _d Gaussian unit 
        vectors of length determined by the second dimension (number of 
        columns) of X
        """
        rng = np.random.default_rng(self._seed)
        n_features = X.shape[1]
        self._hyperplanes = rng.standard_normal(size=(self._d, n_features))

    def transform(self, X: npt.NDArray[np.float64])->npt.NDArray[np.uint8]:
        """
        Project the rows of X into binary vectors
        """
        if self._hyperplanes is None:
            raise ValueError("fit method should be called first")
        projections = np.dot(X, self._hyperplanes.T)
        return (projections >= 0).astype(np.uint8)

    def fit_transform(self, X: npt.NDArray[np.float64])->npt.NDArray[np.uint8]:
        """
        Calls fit() followed by transform()
        """
        self.fit(X)
        return self.transform(X)

class LocalitySensitiveHashing:
    """
    Performs locality-sensitive hashing by projecting unit vectors to binary vectors
    """

    # type hints for intended members
    _D: int # number of random hyperplanes
    _k: int # hash function length
    _L: int # number of hash functions (tables)
    _hash_functions: npt.NDArray[np.int64] # the actual hash functions
    _random_hyperplanes: RandomHyperplanes # random hyperplanes object
    _H: List[Dict[Tuple[np.uint8],Set[int]]] # hash tables
    _X: npt.NDArray[np.float64] # the original data
    
    def __init__(self, D: int, k: int, L: int, seed: Optional[int]):
        """
        Sets the parameters
        - D internal dimensionality (used with random hyperplanes)
        - k length of hash functions (how many elementary hash functions 
          to concatenate)
        - L number of hash tables
        - seed random number generator seed (used for intializing random 
          hyperplanes; also used to seed the random number generator
          for drawing the hash functions)
        """
        self._D = D
        self._k = k
        self._L = L
        rng = np.random.default_rng(seed)
        #raise NotImplementedError()
        # draw the hash functions here
        # (essentially, draw a random matrix of shape L*k with values in
        # 0,1,...,d-1)
        # also initialize the random hyperplanes
        self._hash_functions = rng.integers(0, D, size=(L, k))
        self._random_hyperplanes = RandomHyperplanes(D, self.seed)

    def fit(self, X: npt.NDArray[np.float64]) -> None:
        self._X = X
        # Fit random hyperplanes
        self._random_hyperplanes.fit(X)
        # Project dataset into binary vectors
        binary_vectors = self._random_hyperplanes.transform(X)
        # Hash dataset L times into hash tables
        self._H = [{} for _ in range(self._L)]
        for l in range(self._L):
            for i, binary_vector in enumerate(binary_vectors):
                hash_key = tuple(binary_vector[self._hash_functions[l]])
                if hash_key not in self._H[l]:
                    self._H[l][hash_key] = set()
                self._H[l][hash_key].add(i)

    def query(self, q: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        # Project query into a binary vector
        binary_query = self._random_hyperplanes.transform(q.reshape(1, -1))
        result_indices = set()
        # Hash query L times
        for l in range(self._L):
            hash_key = tuple(binary_query[0][self._hash_functions[l]])
            if hash_key in self._H[l]:
                result_indices.update(self._H[l][hash_key])
        # Compute dot products with those vectors
        dot_products = np.dot(self._X, q)
        # Sort results in descending order and return the indices
        sorted_indices = sorted(result_indices, key=lambda i: dot_products[i], reverse=True)
        return np.array(sorted_indices)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', help='Random hyperplanes dimension', type=int,
                            required = True)
    parser.add_argument('-k', help='Hash function length', type=int,
                            required = True)
    parser.add_argument('-L', help='Number of hash tables (functions)', type=int,
                            required = True)
    parser.add_argument('dataset', help='Glove dataset filename',
                            type=str)
    parser.add_argument('queries', help='Queries filename', type=str)
    args = parser.parse_args()
    
    (word_to_idx, idx_to_word, X) = load_glove(args.dataset)


    X = normalize(X)

    (Q,queries) = construct_queries(args.queries, word_to_idx, X)

    t1 = time.time()
    lsh = LocalitySensitiveHashing(args.D, args.k, args.L, 1234)
    
    t2 = time.time()    
    lsh.fit(X)
    
    t3 = time.time()
    neighbors = list()
    for i in range(Q.shape[0]):
        q = Q[i,:]
        I = lsh.query(q)
        neighbors.append([idx_to_word[i] for i in I][1:4])
    t4 = time.time()

    print('init took',t2-t1)
    print('fit took', t3-t2)
    print('query took', t4-t3)
    print('total',t4-t1)

    for i in range(Q.shape[0]):
        print(f'{queries[i]}: {" ".join(neighbors[i])}') 

