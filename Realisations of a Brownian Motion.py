import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

S0 = 100 # Starting price of the underlying
sigma = 0.2 # Volatility of the underlying
n = 252 # Monitoring dates
T = 1 # Option expiry
r = 0.05 # Risk-free interest rate
M = 100 # Number of simulations
dt = T / n # Time increment of monitoring dates
drift = (r - sigma ** 2 / 2) * dt
# drift = 0

start_time = time.time()

St = np.zeros(n + 1)  # +1 to include S0 in the count
St[0] = S0  # Set the first element to S0
pathes = np.zeros((M, n + 1))
pathes[:, 0] = S0

for j in range(0, M):
    for i in range(1, n + 1):
        St[i] = St[i-1] * np.exp(drift + sigma * np.sqrt(dt) * np.random.randn())
        pathes[j, i] = St[i]  # Assign the computed value to the path matrix

pathes_df = pd.DataFrame(pathes)

# Parameters
dpi_value = 600
line_width = 0.75
alpha_value = 0.5
figure_size = (10, 6)

# Generate time points
time_points = np.linspace(0, T ,n+1)

# Plot
plt.figure(figsize=figure_size, dpi=dpi_value)
for j in range(M):
    plt.plot(time_points, pathes_df.iloc[j, :], lw=line_width, alpha=alpha_value)

plt.title(f'{M} Realisations of a Geometric Brownian Motion.png')
plt.xlabel('t')
plt.ylabel('S(t)')
plt.show()

end_time = time.time()
execution_time = end_time - start_time
print(f"Total Time Elapsed: {execution_time:.1f} seconds")
