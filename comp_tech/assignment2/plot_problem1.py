import matplotlib.pyplot as plt
import numpy as np
# Given data
cores = [1, 2, 4, 8, 16, 32, 40, 50, 64]
parallel_time = [71.06996083259583, 59.798534631729126, 29.952446460723877, 14.844865798950195,
             9.051558017730713, 4.3613975048065186, 3.6086201667785645, 2.596949577331543,
             2.3571126461029053]
sequential_time = [0.011720657348632812,0.013322830200195312,
             0.015375137329101562,0.022853851318359375
             ,0.03952527046203613,0.05037498474121094
             ,0.06037473678588867,0.07226753234863281
             ,0.09207868576049805]

base_time = parallel_time[0] + sequential_time[0]
# Calculate observed speedup
observed_speedup = np.zeros(len(parallel_time))
for i, val in enumerate(parallel_time):
    observed_speedup[i] = base_time/(val + sequential_time[i])
# Upper bound on speedup

p = parallel_time[0]/(sequential_time[0]+parallel_time[0])

amdahls_lim = 1/(1-p)
upper_bound_speedup = [amdahls_lim for _ in cores]

# Amdahl's Law
theoretical_speedup = [1 / ((1 - p) + (p / n)) for n in cores]



# Plot
plt.figure(figsize=(10, 6))
plt.plot(cores, observed_speedup, marker='o', label='Observed Speedup')
plt.plot(cores, theoretical_speedup, marker='x', label='Theoretical Speedup (Amdahl\'s Law)')
plt.plot(cores, upper_bound_speedup, linestyle='--', label='Upper Bound (Amdahl\'s Law)')
plt.xlabel('Number of CPU Cores')
plt.ylabel('Speedup')
plt.title('Speedup vs. Number of CPU Cores')
plt.legend()
plt.grid(True)
plt.savefig("Speedup_without_Limit.pdf")
plt.show()
