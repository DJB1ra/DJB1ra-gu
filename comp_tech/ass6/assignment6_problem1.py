#!/usr/bin/env python3

import time
import argparse
import findspark
findspark.init()
from pyspark import SparkContext
import numpy as np


def f(y: np.uint64) -> np.uint64:
    """
    Multiply-shift hash function to select the index for the substream
    """
    #same f as in assignment5
    product = np.uint64(np.uint64(0xc863b1e7d63f37a3) * y)
    shifter = np.uint64(64) - np.uint64(np.log2(m))
    return product >> shifter

def p(num: np.uint32) -> np.uint16:
    #same rho as in assignment5
    mask = np.uint64(1) << np.uint64(32)
    position = np.uint16(0)

    while mask:
        if num & mask:
            return position
        mask >>= np.uint64(1)
        position += np.uint16(1)
    return position

def tabulation_hash(x: np.uint64, tables) -> np.uint32:
    #essentially the same as in assignment 5
    hash_value = np.uint32(0)
    for i in range(4):
        chunk = (x >> (np.uint64(i) * np.uint64(16))) & np.uint64(0xFFFF)
        hash_value ^= tables[i][chunk]
    return hash_value

def process_line(line, tables_bc):
    user_ids = line.split()  #split the ids
    hash_values = []
    for user_id in user_ids: #hash both ids
        user_id = np.uint64(user_id)
        x = tabulation_hash(user_id, tables_bc.value)
        j = f(user_id)
        hash_values.append((j, p(x)))  #and append the hashed user IDs
    return hash_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Twitter followers.')
    parser.add_argument('-w', '--num-workers', default=1, type=int,
                        help='Number of workers')
    parser.add_argument('filename', type=str, help='Input filename')
    args = parser.parse_args()

    start = time.time()
    sc = SparkContext(master=f'local[{args.num_workers}]')

    lines = sc.textFile(args.filename)

    m = 1024
    seed = 69
    np.random.seed(seed)

    tables = []
    for _ in range(4):  
        table = np.random.randint(0, 2**32, size=65536, dtype=np.uint32)
        tables.append(table)

    #put same table to every worker
    tables_bc = sc.broadcast(tables)

    # Map step: Process each line to get hash values
    hash_values = lines.flatMap(lambda line: process_line(line, tables_bc))

    # Reduce step: Aggregate the results to update registers using reduceByKey
    reduced_registers_rdd = hash_values.map(lambda x: (x[0], x[1]))
    reduced_registers = reduced_registers_rdd.reduceByKey(max).collectAsMap()

    # Prepare the register array for cardinality estimation
    registers = np.zeros(m, dtype=int)
    for j, position in reduced_registers.items():
        registers[j] = position

    # Calculate the cardinality estimate using the provided method
    m = len(registers)
    alpha = 0.7213 / (1 + (1.079 / m))
    harm = np.sum(1 / (2 ** registers))
    cardinality_estimate = alpha * m ** 2 * (1 / harm)

    # Bias correction for small ranges
    zeroes = np.count_nonzero(registers == 0)
    if cardinality_estimate <= (2.5 * m) and zeroes > 0:
        cardinality_estimate = m * np.log(m / zeroes)
    elif cardinality_estimate > (np.power(2, 32) / 30):
        cardinality_estimate = np.power(2, 32) * np.log(1 - (cardinality_estimate / np.power(2, 32)))

    end = time.time()

    # Print the estimated number of users and other relevant information
    print(f'estimated number of users: {int(cardinality_estimate)}')
    print(f'num workers: {args.num_workers}')
    print("Time taken:", end-start)
