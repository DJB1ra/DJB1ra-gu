#!/usr/bin/env python3

import time
import argparse
import findspark
findspark.init()
from pyspark import SparkContext

# Function to read the lines
def parse_line(line):
    parts = line.split(':')
    user_id = int(parts[0])
    follows = [] if len(parts) == 1 else list(map(int, parts[1].split()))
    return user_id, follows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Twitter followers.')
    parser.add_argument('-w', '--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('filename', type=str, help='Input filename')
    args = parser.parse_args()

    start = time.time()
    sc = SparkContext(master=f'local[{args.num_workers}]')

    lines = sc.textFile(args.filename)
    parsed_lines = lines.map(parse_line)
    num_users = parsed_lines.count()

    #transform key, value to key, 1. 
    followers_count = parsed_lines.flatMap(lambda x: [(follower, 1) for follower in x[1]]) \
                                  .reduceByKey(lambda a, b: a + b)

    # Calculate maximum number of followers and corresponding user ID
    max_followers = followers_count.reduce(lambda x, y: x if x[1] > y[1] else y)

    #avg followers
    total_followers = followers_count.map(lambda x: x[1]).sum()
    average_followers = total_followers / num_users

    #filter out no followers
    no_followers_count = parsed_lines.filter(lambda x: len(x[1]) == 0).count()

    end = time.time()
    total_time = end - start

    # Print results
    print(f'max followers: {max_followers[0]} has {max_followers[1]} followers')
    print(f'followers on average: {average_followers}')
    print(f'number of users with no followers: {no_followers_count}')
    print(f'num workers: {args.num_workers}')
    print(f'total time: {total_time}')

