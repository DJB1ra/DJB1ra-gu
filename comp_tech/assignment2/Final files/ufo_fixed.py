#!/usr/bin/env python3

from typing import List, Tuple
import pandas as pd # type: ignore
import argparse
import sys
from sklearn.metrics.pairwise import haversine_distances # type: ignore
from itertools import combinations
import numpy as np
from math import sin, cos, sqrt, asin
import numpy.typing as npt
import random
from operator import itemgetter
import time

# this should be replaced with the multiprocessing variant
from multiprocessing import Queue, Process, shared_memory
import multiprocessing as mp #just so we dont mix up the "normal" array with multprocess



def haversine(lat1: float, lon1: float, lat2: float, lon2: float)->float:
    """
    Computes haversine distance manually.
    
    Parameters:
    - lat1: latitude of point 1
    - lon1: longitude of point 1
    - lat2: latitude of point 2
    - lon2: longitude of point 2
    """
    return 2*asin(sqrt( sin((lat2-lat1)/2)**2 +
                            cos(lat1)*cos(lat2) * sin((lon2-lon1)/2)**2))




def find_nearest(X: npt.NDArray[np.float64], q: npt.NDArray)->int:
    """
    Find the index of the nearest point in X, with respect to a query point q.

    Parameters:
    - X: (n,2) dataset of (latitude,longitude) pairs
    - q: a (latitude,longitude) query point

    Return value:
    index i such that X[i,:] has minimal haversine distance wrt. q
    """
    D = haversine_distances(X,q.reshape(1,2))
    return int(np.argmin(D))




# TODO: change the type of A to be a multiprocessing array

def process(Q: mp.Queue, R: mp.Queue, A: mp.Array)->None:
    """
    Read one item at a time from the queue, until None is encountered,
    and query the array of latitude/longitude pairs for the nearest index

    Parameters:
    - Q: queue of (idx,latitude,longitude) pairs, until None is encountered
    - R: queue of (idx,i) pairs that contain the indices of nearest neighbors, 
         and ends with a None
    - A: an n*2 array of latitude longitude pairs
    """

    X: npt.NDArray[np.float64] = np.frombuffer(A.get_obj(), dtype=np.float64).reshape((-1,2))

    while True:
        q = Q.get()
        if q is None:
            break
        idx = q[0]
        q = np.array(q[1:])
        R.put((idx,find_nearest(X,q)))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'Ufo',
        description = 'Use linear scan to detect the nearest UFO sighitings '
        'by the latitude and longitude, with respect to the haversine '
        'distance. Queries are read from stdin, one query per line, given '
        'in degrees.'
        )
    parser.add_argument('filename', type = str, default = 'ufo.csv',
                            help = 'database filename, should point to the '
                            '"ufo.csv" file',
                            nargs = '?')
    parser.add_argument('-w', '--workers', type = int, default = 1,
                            metavar = 'W',
                            help = 'Number of workers (processes)')
    parser.add_argument('-q', '--queries', type = str, default = None,
                            help = 'Read queries from this file (default: stdin')

    args = parser.parse_args()
    t1 = time.time()

    # read data
    df = pd.read_csv(args.filename)
    n: int = df.shape[0]
    locs: List[str] = df['location'].tolist()
    print('Data read from file')

    # Memory allocation in the form of a multiprocessing array
    # Allocate shared memory
    A = mp.Array('d', n * 2)


    # underlying memory; change this to point to the memory in the
    # multiprocessing array (using np.frombuffer)
    
    # Create a NumPy array that shares memory with the multiprocessing array
    X = np.frombuffer(A.get_obj(), dtype=np.float64).reshape((-1, 2))

    # convert to radians
    # note: Pandas stores data in Fortran order, we convert it to C order
    X[:,0] = df['latitude'] * (np.pi/180)
    X[:,1] = df['longitude'] * (np.pi/180)
    print('Data to rad')

    # this queue contains items to be consumed
    Q = mp.Queue()

    # this queue contains the results
    R = mp.Queue()


    t2 = time.time()  
    workers = [Process(target=process, args=(Q, R, A)) for _ in range(args.workers)]

    for w in workers:
        w.start()

    i = 0
    with open(args.queries,'r') if args.queries is not None else \
      sys.stdin as f:
      for line in f:
        q: npt.NDArray[np.float64] = np.array(list(map(float,line.split()))) \
          * (np.pi/180)
        Q.put((i,q[0],q[1]))
        i += 1
    t3 = time.time()
        
    for _ in range(args.workers):
        Q.put(None)

    res = [R.get() for _ in range(i)]

    for w in workers:
        w.join()
    t4 = time.time()
   
    # sort by the index so locations are printed in correct order
    res.sort()
    for (_,i) in res:
        print(locs[i])
    t5 = time.time()

    tot_time = t5 - t1
    par_time = t4- t2
    seq_time = tot_time - par_time


    print('workers', args.workers)
    print('total time',tot_time)
    print('parallel time',par_time)
    print('sequential time',seq_time)
    print('parallel proportion',par_time/tot_time)
    print('Sequential proportion',seq_time/tot_time)