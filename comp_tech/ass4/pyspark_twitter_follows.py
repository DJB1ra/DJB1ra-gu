#!/usr/bin/env python3

import time
import argparse
import findspark
findspark.init()
from pyspark import SparkContext

#function to read the lines
def parse_line(line):
    parts = line.split(':')
    user_id = int(parts[0])
    follows = [] if len(parts) == 1 else list(map(int, parts[1].split()))
    return user_id, follows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Twitter follows.')
    parser.add_argument('-w', '--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('filename', type=str, help='Input filename')
    args = parser.parse_args()

    start = time.time()
    sc = SparkContext(master=f'local[{args.num_workers}]')

    lines = sc.textFile(args.filename)
    parsed_lines = lines.map(parse_line)

    #num of users (count the lines in the document)
    num_users = parsed_lines.count()

    #for each id (index 0), we count the length of index 1 (after the :)
    #Map since we want to iterate over all ID:s 
    follows_count = parsed_lines.map(lambda x: (x[0], len(x[1])))

    #a check for max follows, assign the next max follows (y) if its larger than x
    #Reduce since we have two arguments
    max_follows = follows_count.reduce(lambda x, y: x if x[1] >= y[1] else y)
    max_id, max_follow_count = max_follows

    #sum up all the follows
    total_follows = follows_count.map(lambda x: x[1]).sum()
    average_follows = total_follows / num_users

    #filter out users that doesnt follow anyone
    no_followers_count = follows_count.filter(lambda x: x[1] == 0).count()

    end = time.time()
    total_time = end - start

    # Output statistics
    print(f'max follows: {max_id} follows {max_follow_count}')
    print(f'users follow on average: {average_follows}')
    print(f'number of users who follow no-one: {no_followers_count}')
    print(f'num workers: {args.num_workers}')
    print(f'total time: {total_time}')

