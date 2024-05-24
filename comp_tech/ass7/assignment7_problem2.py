import cupy as cp
import argparse
import pandas as pd
import csv
import sys
import time

def linear_scan(X, Q, b=None):
    """
    Perform linear scan for querying nearest neighbor with batch processing on GPU.
    X: n*d dataset
    Q: m*d queries
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
    for batch in range(num_batches):
        start = batch * b
        end = min(start + b, m)
        Q_batch = Q[start:end]

        # Compute distances
        diff = X[cp.newaxis, :, :] - Q_batch[:, cp.newaxis, :]
        dist = cp.linalg.norm(diff, axis=2)
        
        # Find the indices of the minimum distances
        I[start:end] = cp.argmin(dist, axis=1)
    
    return I

def load_glove(fn):
    """
    Loads the glove dataset from the file
    Returns (X,L) where X is the dataset vectors and L is the words associated
    with the respective rows.
    """
    df = pd.read_table(fn, sep=' ', index_col=0, header=None,
                       quoting=csv.QUOTE_NONE, keep_default_na=False)
    X = cp.asarray(df.values, dtype=cp.float32)
    L = df.index.tolist()
    return (X, L)

def load_pubs(fn):
    """
    Loads the pubs dataset from the file
    Returns (X,L) where X is the dataset vectors (easting,northing) and 
    L is the list of names of pubs, associated with each row
    """
    df = pd.read_csv(fn)
    L = df['name'].tolist()
    X = cp.asarray(df[['easting', 'northing']].values, dtype=cp.float32)
    return (X, L)

def load_queries(fn):
    """
    Loads the m*d array of query vectors from the file
    """
    return cp.loadtxt(fn, delimiter=' ', dtype=cp.float32)

def load_query_labels(fn):
    """
    Loads the m-long list of correct query labels from a file
    """
    with open(fn, 'r', encoding='utf-8') as f:
        return f.read().splitlines()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform nearest neighbor queries under the '
                    'Euclidean metric using linear scan on GPU, measure the time '
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

    assert cp.ascontiguousarray(X).flags['C_CONTIGUOUS']
    assert cp.ascontiguousarray(Q).flags['C_CONTIGUOUS']
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

     #Record transfer time for dataset
    dataset_transfer_start = time.time()
    X_gpu = cp.asarray(X, cp.float32, order='C', blocking=True)
    cp.cuda.Stream.null.synchronize()  # synchronize to make sure the operation is completed
    dataset_transfer_end = time.time()
    dataset_transfer_time = (dataset_transfer_end - dataset_transfer_start)  

    # Record transfer time for queries
    queries_transfer_start = time.time()
    Q_gpu = cp.asarray(Q, cp.float32, order='C', blocking=True)
    cp.cuda.Stream.null.synchronize()  # synchronize to make sure the operation is completed
    queries_transfer_end = time.time()
    queries_transfer_time = (queries_transfer_end - queries_transfer_start)  

    t7 = time.time()
    I_gpu = linear_scan(X_gpu, Q_gpu, args.batch_size)
    cp.cuda.Stream.null.synchronize()  # synchronize to make sure the operation is completed
    t8 = time.time()

    # Transfer results back to the host and record transfer time
    results_transfer_start = time.time()
    I = cp.asnumpy(I_gpu, order='C', blocking=True)
    cp.cuda.Stream.null.synchronize()  # synchronize to make sure the operation is completed
    results_transfer_end = time.time()
    results_transfer_time = (results_transfer_end - results_transfer_start)  

    # Calculate throughput
    total_query_time = t8 - t7
    throughput = m / total_query_time if total_query_time > 0 else float('inf')

    num_erroneous = 0
    if QL is not None:
        for i, j in enumerate(I):
            if QL[i] != L[j]:
                sys.stderr.write(f'{i}th query was erroneous: got "{L[j]}", '
                                 f'but expected "{QL[i]}"\n')
                num_erroneous += 1

    print(f'Loading dataset ({n} vectors of length {d}) took', t2-t1)
    print(f'Dataset transfer time: {dataset_transfer_time} s')
    print(f'Loading queries took', t4-t3)
    print(f'Queries transfer time: {queries_transfer_time} s')
    print(f'Loading labels took', t6-t5)
    print(f'Performing {m} NN queries took', t8-t7)
    print(f'Results transfer time: {results_transfer_time} s')
    print(f'Number of erroneous queries: {num_erroneous}')
    print(f'Throughput: {throughput:.2f} queries per second')