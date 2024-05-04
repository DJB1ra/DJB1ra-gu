from pyspark import SparkContext
import argparse
import re

def extract_user(line):
    # Use regular expression to extract user account
    match = re.search(r'for\s(?:invalid user\s)?(\w+)\sfrom', line)
    if match:
        return match.group(1)
    else:
        return None

def extract_repeated(line):
    # Use regular expression to check for repeated message and extract count
    match = re.search(r'message repeated (\d+) times', line)
    if match:
        return int(match.group(1))
    else:
        return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LogParser.')
    parser.add_argument('filename', type=str, help='Input filename')
    args = parser.parse_args()

    sc = SparkContext("local", "LogAnalysis")

    # Read the log file
    lines = sc.textFile(args.filename)

    # Filter lines to get all attempts containing "password"
    password_attempts = lines.filter(lambda x: 'password' in x)

    # Count the number of lines containing 'password'
    password_count = password_attempts.count()

    print("Number of lines containing 'password':", password_count)

    # Filter lines to get successful and unsuccessful attempts
    successful_attempts = password_attempts.filter(lambda x: 'Accepted password' in x)
    unsuccessful_attempts = password_attempts.filter(lambda x: 'Failed password' in x)
    repeated_attempts = password_attempts.filter(lambda x: 'message repeated' in x)

    # Extract user account from each line for successful attempts
    successful_attempts_count = successful_attempts.map(lambda x: (extract_user(x), 1))
    # Extract user account from each line for unsuccessful attempts
    unsuccessful_attempts_count = unsuccessful_attempts.map(lambda x: (extract_user(x), 1))

    # Count the occurrences of each user account for successful attempts
    successful_user_counts = successful_attempts_count.reduceByKey(lambda x, y: x + y)
    # Count the occurrences of each user account for unsuccessful attempts
    unsuccessful_user_counts = unsuccessful_attempts_count.reduceByKey(lambda x, y: x + y)

    # Combine counts for successful and unsuccessful attempts for each user
    combined_user_counts = successful_user_counts.union(unsuccessful_user_counts)

    # Extract user and repetition count from repeated logs
    repeated_user_counts = repeated_attempts.map(lambda x: (extract_user(x), extract_repeated(x)))

    # Aggregate total counts for repeated attempts
    total_repeated_counts = repeated_user_counts.reduceByKey(lambda x, y: x + y)

    # Union counts for regular and repeated attempts
    all_user_counts = combined_user_counts.union(total_repeated_counts)

    # Find the user account with the highest count
    max_attempts_user = all_user_counts.max(lambda x: x[1])

    print("User with the largest number of login attempts:", max_attempts_user[0])
    print("Number of attempts:", max_attempts_user[1])
