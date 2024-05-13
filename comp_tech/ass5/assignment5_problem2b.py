import numpy as np
import numpy.typing as npt
from typing import Optional, Callable
from assignment5_problem1 import TabulationHash
import time

class HyperLogLog:
    _h: TabulationHash
    _f: Callable[[np.uint64], np.uint64]
    _M: npt.NDArray[np.uint8]
    _a: np.uint64

    def __init__(self, m: int, seed: Optional[int] = None):
        """
        Initialize a HyperLogLog sketch
        """
        self._M = np.zeros(m, dtype=np.uint8)
        self._h = TabulationHash(seed=seed)
        self._a = np.uint64(0xc863b1e7d63f37a3)
    
    def _p(self, num: np.uint32):
        mask = np.uint64(1) << np.uint64(32)
        position = np.uint16(0)

        while mask:
            if num & mask:
                return position
            mask >>= np.uint(1)
            position += np.uint16(1)
        return position

    
    def _f(self, y: np.uint64) -> np.uint64:
        """
        Multiply-shift hash function to select the index for the substream
        """
        shift_amount = np.uint64(64 - int(np.log2(len(self._M))))
        product = np.uint64(self._a * y)
        return (product >> shift_amount)


    def __call__(self, x: np.uint64):
        """
        Add element into the sketch
        """
        j = self._f(x)
        if j >= len(self._M):
            raise IndexError("Index j exceeds the length of _M array")
        hash_value = self._h(x)
        position = self._p(hash_value)
        self._M[j] = max(int(self._M[j]), int(position))

    def merge(self, other: 'HyperLogLog') -> 'HyperLogLog':
        """
        Merge two sketches
        """
        if len(self._M) != len(other._M):
            raise ValueError("Sketches must have the same number of registers for merging.")
        self._M = np.maximum(self._M, other._M)
        return self
        
    def estimate(self)->float:
            """
            Return the present cardinality estimate
            """
            m = len(self._M)
            alpha = 0.7213 / (1 + (1.079 / m))
            harm = []
            for i in self._M:
                harm.append(1/2**i)
                #print("M", i)
            harm = 1/(np.sum(harm))

            cardinality_estimate = alpha * m**2 * harm

            zeroes = np.count_nonzero(self._M == 0)

            if cardinality_estimate <= (2.5 * m) and zeroes > 0:
                print("2.5")
                cardinality_estimate = 0
                for i in range(len(self._M)):
                    cardinality_estimate += self._M[i]
            elif cardinality_estimate > (np.power(2,32)/30):
                print("2**32")
                cardinality_estimate = np.power(2, 32) * np.log(1 - (cardinality_estimate / np.power(2, 32)))

            return cardinality_estimate
        

if __name__ == '__main__':
    obj = HyperLogLog(1024, seed=420)
    numbers = np.arange(1, 1000000, dtype=np.uint64)
    

    for x in numbers:
        obj(x)
    
    print(obj.estimate())
