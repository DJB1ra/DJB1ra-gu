#!/bin/bash

echo "$(lscpu|grep -E "Model name")"
echo "CPU freq: $(grep -m1 'cpu MHz' /proc/cpuinfo)"


sockets=$(lscpu | grep "Socket(s)"|awk '{print $2}')
cores_per_socket=$(lscpu | grep "Core(s) per socket"|awk '{print $4}')
threads_per_core=$(lscpu | grep "Thread(s) per core"|awk '{print $4}')

total_cores=$((sockets*cores_per_socket))
total_threads=$((total_cores*threads_per_core))

echo "Sockets: $sockets"
echo "Total cores: $total_cores"
echo "Total threads: $total_threads"

echo "Architecture:"
echo "$(lscpu | grep "Architecture" | awk '{print $2}')"

echo "Cache size:"
echo "$(cat /proc/cpuinfo | grep -m2 'cache size\|cache_alignment')"

echo "L1-3 cache:"
lscpu | grep -E 'cache:'

#it exists different ways to access the amount of ram but 
#the other commands gave way more or less RAM than
#specified on the Canvas page
echo "System RAM:"
echo "$(free -h --giga | grep 'Mem:'|awk '{print$2}')"

echo "Number(s), model of GPU and RAM:"
nvidia-smi -q | grep "Product Name" | awk '{print $4, $5}'
nvidia-smi -q | grep "Total" -m1 | awk '{print $3, $4}'

echo "Filesystem:"
df -Th | grep "data"


echo "Linux kernel version:"
uname -r

echo "Linux distribution/version:"
lsb_release -a | grep "Description" | awk '{print $2, $3}'

echo "Python file:"
which python3

echo "Python version:"
python3 --version
