import cupy as cp
import numpy as np
import argparse
import pandas as pd
import csv
import sys
import time

def efficient_linear_scan_gpu(X, Q, b=None):
    """
    Perform linear scan for querying nearest neighbor with batch processing using GPU.
    This method uses CuPy for GPU acceleration to compute distances more efficiently.
    X: n*d dataset (CuPy array)
    Q: m*d queries (CuPy array)
    b: optional batch size
    Returns an m-vector of indices I; the value i reports the row in X such 
    that the Euclidean norm of ||X[I[i],:]-Q[i]|| is minimal
    """
    n, d = X.shape
    m = Q.shape[0]
    I = cp.zeros(m, dtype=cp.int64)
    
    if b is None or b <= 0:
        b = m
    
    num_batches = (m + b - 1) // b
    
    # Precompute the squared norms of the dataset vectors
    X_squared_norms = cp.sum(X**2, axis=1)
    
    for batch in range(num_batches):
        start = batch * b
        end = min(start + b, m)
        Q_batch = Q[start:end]

        # Compute the squared norms of the query vectors
        Q_batch_squared_norms = cp.sum(Q_batch**2, axis=1)

        # Compute the dot product between X and Q_batch.T
        dot_product = cp.dot(X, Q_batch.T)

        # Compute the distance matrix using the formula
        dists = X_squared_norms[:, cp.newaxis] + Q_batch_squared_norms[cp.newaxis, :] - 2 * dot_product

        # Find the indices of the minimum distances
        I[start:end] = cp.argmin(dists, axis=0)
    
    return I.get()  # Transfer the result back to the host

def load_glove(fn):
    """
    Loads the glove dataset from the file
    Returns (X,L) where X is the dataset vectors and L is the words associated
    with the respective rows.
    """
    df = pd.read_table(fn, sep=' ', index_col=0, header=None,
                       quoting=csv.QUOTE_NONE, keep_default_na=False)
    X = np.ascontiguousarray(df, dtype=np.float32)
    L = df.index.tolist()
    return (cp.asarray(X), L)

def load_pubs(fn):
    """
    Loads the pubs dataset from the file
    Returns (X,L) where X is the dataset vectors (easting,northing) and 
    L is the list of names of pubs, associated with each row
    """
    df = pd.read_csv(fn)
    L = df['name'].tolist()
    X = np.ascontiguousarray(df[['easting', 'northing']], dtype=np.float32)
    return (cp.asarray(X), L)

def load_queries(fn):
    """
    Loads the m*d array of query vectors from the file
    """
    return cp.loadtxt(fn, delimiter=' ', dtype=cp.float32)

def load_query_labels(fn):
    """
    Loads the m-long list of correct query labels from a file
    """
    with open(fn, 'r') as f:
        return f.read().splitlines()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform nearest neighbor queries under the '
                    'Euclidean metric using linear scan with GPU acceleration, measure the time '
                    'and optionally verify the correctness of the results')
    parser.add_argument(
        '-d', '--dataset', type=str, required=True,
        help='Dataset file (must be pubs or glove)')
    parser.add_argument(
        '-q', '--queries', type=str, required=True,
        help='Queries file (must be compatible with the dataset)')
    parser.add_argument(
        '-l', '--labels', type=str, required=False,
        help='Optional correct query labels; if provided, the correctness '
             'of returned results is checked')
    parser.add_argument(
        '-b', '--batch-size', type=int, required=False,
        help='Size of batches')
    args = parser.parse_args()

    t1 = time.time()
    if 'pubs' in args.dataset:
        (X, L) = load_pubs(args.dataset)
    elif 'glove' in args.dataset:
        (X, L) = load_glove(args.dataset)
    else:
        sys.stderr.write(f'{sys.argv[0]}: error: Only glove/pubs supported\n')
        exit(1)
    t2 = time.time()

    (n, d) = X.shape
    assert len(L) == n

    t3 = time.time()
    Q = load_queries(args.queries)
    t4 = time.time()

    assert X.flags['C_CONTIGUOUS']
    assert Q.flags['C_CONTIGUOUS']
    assert X.dtype == cp.float32
    assert Q.dtype == cp.float32

    m = Q.shape[0]
    assert Q.shape[1] == d

    t5 = time.time()
    QL = None
    if args.labels is not None:
        QL = load_query_labels(args.labels)
        assert len(QL) == m
    t6 = time.time()

    I = efficient_linear_scan_gpu(X, Q, args.batch_size)
    t7 = time.time()
    assert I.shape == (m,)

    num_erroneous = 0
    if QL is not None:
        for i, j in enumerate(I):
            if QL[i] != L[j]:
                sys.stderr.write(f'{i}th query was erroneous: got "{L[j]}", '
                                 f'but expected "{QL[i]}"\n')
                num_erroneous += 1

    print(f'Loading dataset ({n} vectors of length {d}) took', t2-t1)
    print(f'Loading queries ({m} vectors of length {d}) took', t4-t3)
    print(f'Loading query labels took', t6-t5)
    print(f'Performing {m} NN queries took', t7-t6)
    print(f'Number of erroneous queries: {num_erroneous}')