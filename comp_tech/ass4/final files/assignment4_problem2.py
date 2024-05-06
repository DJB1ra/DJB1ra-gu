#!/usr/bin/env python3

import argparse
import time
import re
import findspark
findspark.init()
from pyspark import SparkContext

def handle_repeated_attempts(line):
    #unpack repeated logs
    repeat_match = re.search(r'message repeated (\d+) times:', line)
    if repeat_match:
        message_repeat_count = int(repeat_match.group(1))
        return [line] * message_repeat_count
    else:
        return [line]
    
#function to extract the username, two cases with "for" and "invalid user"
def extract_username(line):
    username = None

    #check for invalid user first since otherwise it will just get stuck on for with 
    #the repeated messages
    if "invalid user" in line:
        username_index = line.split().index("user") + 1
        username = line.split()[username_index]
    elif "for" in line:
        username_index = line.split().index("for") + 1
        username = line.split()[username_index]

    return (username, 1)

#ip address extractor
def extract_ip(line):
    #regular expression for the ip address
    ip_extract = re.compile(r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}')
    
    ip_match = ip_extract.search(line)
    #check if ip address isin the log, otherwise its None
    ip_address = ip_match.group(0) if ip_match else None

    return (ip_address, 1)

#extract date
def extract_date(line):
    parts = line.split() 
    month = parts[0] 
    day = parts[1] 
    return (month, day)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LogParser.')
    parser.add_argument('filename', type=str, help='Input filename')
    parser.add_argument('-w', '--num-workers', default = 1, type = int, help = 'Number of workers')
    args = parser.parse_args()

    start = time.time()

    spark_context = SparkContext(master = f'local[{args.num_workers}]')

    #read file
    lines = spark_context.textFile(args.filename)

    #first filter on if line contains "password" 
    password_attempts = lines.filter(lambda x: 'password' in x).flatMap(handle_repeated_attempts).cache()


    #filter again from the filtered logs based on accepted and failed password. Also for invalid users
    successful_attempts = password_attempts.filter(lambda x: 'Accepted password' in x).cache()

    unsuccessful_attempts = password_attempts.filter(lambda x: 'Failed password' in x).cache()

    invalid_user_attempts = unsuccessful_attempts.filter(lambda x: 'invalid user' in x).cache()

    #rdd which contains all logs with "password" in them. use extract_username function
    #and then map each username to a value (in this case 1). Then reduce by the username
    #which counts all the ones(1).
    all_attempts_count = password_attempts.map(lambda x: extract_username(x)) \
                                                .map(lambda input: (input, input[1])) \
                                                .reduceByKey(lambda x, y: x + y).cache()

    successful_attempts_count = successful_attempts.map(lambda x: extract_username(x)) \
                                                .map(lambda input: (input, input[1])) \
                                                .reduceByKey(lambda x, y: x + y).cache()

    unsuccessful_attempts_count = unsuccessful_attempts.map(lambda x: extract_username(x))\
                                                .map(lambda input: (input, input[1]))\
                                                .reduceByKey(lambda x, y: x + y).cache()

    invalid_user_attempts_count = invalid_user_attempts.map(lambda x: extract_username(x))\
                                                .map(lambda input: (input, input[1]))\
                                                .reduceByKey(lambda x, y: x + y).cache()
    
    
    #task a
    (max_user_login_attempts, max_count) = all_attempts_count.max(key=lambda x: x[1])

    #task b 
    (max_success_user_attempts, max_success_count) = successful_attempts_count.max(key=lambda x: x[1])
    
    #task c
    (max_unsuccess_user_attempts, max_unsuccess_count) = unsuccessful_attempts_count.max(key=lambda x: x[1])
    
    #task d
    success_rates = all_attempts_count.join(successful_attempts_count) \
                                    .mapValues(lambda x: x[1] / x[0])

    #max success rate
    max_success_rate = success_rates.map(lambda x: x[1]).max()

    #find the other users with the same max success rate
    users_with_max_success_rate = success_rates.filter(lambda x: x[1] == max_success_rate) \
                                            .collect()

    #task e
    #same methodology as extractions of username, but instead we have the ip as key.
    ip_counts = unsuccessful_attempts.map(lambda x: extract_ip(x))\
                            .map(lambda x: (x))\
                            .reduceByKey(lambda x, y: x + y)\
                            .sortBy(lambda x: x[1], ascending=False)  # Sort IP addresses by count in descending order

    #top 3 ip with take (basically the head)
    top_3_ips = ip_counts.take(3)

    #task f
    #takeordereed is descending order (max at the top)
    top_10_invalid_user_attempts = invalid_user_attempts_count.takeOrdered(10, key=lambda x: -x[1])

    #task g
    date_counts = password_attempts.map(lambda x: extract_date(x)) \
    .map(lambda date: ((date[0], date[1]), 1)) \
    .reduceByKey(lambda x, y: x + y) \
    .collect()
    
    #max and min value for the key, value pair of date and count
    max_day = max(date_counts, key=lambda x: x[1])
    min_day = min(date_counts, key=lambda x: x[1])

    end = time.time()

    total_time = end - start

    print(f'Most login attempts by: {max_user_login_attempts[0]}, {max_count} times')
    print(f'Most successful login attempts by: {max_success_user_attempts[0]}, {max_success_count} times')
    print(f'Most unsuccessful login attempts by: {max_unsuccess_user_attempts[0]}, {max_unsuccess_count} times')
    print(f'User accounts with the highest success rate of logins: {users_with_max_success_rate}')
    print(f'Top 3 ip addresses_ {top_3_ips}')
    print("Top 10 invalid user attempts:")
    for user, count in top_10_invalid_user_attempts:
        print(f"User: {user[0]}, count: {count}")
    print(f'Day with the most login activity: {max_day}')
    print(f'Day with the least login activity: {min_day}')
    print('Time taken:', total_time)