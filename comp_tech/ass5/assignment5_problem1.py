#!/usr/bin/env python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import Optional

class TabulationHash:
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the tabulation hash function.
        """
        self.table = np.zeros((4, 65536), dtype=np.uint32)  
        if seed is not None:
            np.random.seed(seed)
        for i in range(4):
            self.table[i] = np.random.randint(0, 2**32, size=65536, dtype=np.uint32)

    def __call__(self, x: np.uint64) -> np.uint32:
        """
        Hash a 64-bit integer key x into a 32-bit hash value.
        """
        h = 0
        x = int(x)
        for i in range(8):
            byte = (x >> (i * 16)) & 0xFFFF  # Using 16-bit alphabet
            h ^= self.table[i % 4, byte]
        return np.uint32(h)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='hashing',
        description='Hash unsigned 64-bit integer keys.',
    )
    parser.add_argument('--seed', type=int, required=False,
                        help='set the seed for the hash function')

    args = parser.parse_args()

    hash_func = TabulationHash(seed=args.seed)

    # Hash integers from 1 to 1,000,000
    hash_values = [hash_func(np.uint64(i)) for i in range(1, 1000001)]

    # Plot histogram of hash values
    plt.hist(hash_values, bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Hash Values')
    plt.xlabel('Hash Value')
    plt.ylabel('Frequency')
    plt.show()
