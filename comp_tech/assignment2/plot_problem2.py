import matplotlib.pyplot as plt
import numpy as np
# Given data
cores = [1, 2, 4, 8, 16, 32, 40, 50, 64]
parallel_time = [63.68037223815918,44.830562353134155,
                 22.73331308364868,11.366667032241821,
                 6.073186159133911,3.2681033611297607,
                 2.7216014862060547,2.335402488708496,
                 1.94974684715271]
sequential_time = [0.09009981155395508,0.0901343822479248,
                   0.09004902839660645,0.09095525741577148,
                   0.09375238418579102,0.09122824668884277,
                   0.09448885917663574,0.0924534797668457,
                   0.09200048446655273]

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
