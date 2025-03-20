#remember:
#!pip install brian2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brian2 import *

start_scope()

# Resets the Brian2 simulation environment.
# Brian2 keeps track of all objects (neurons, monitors, etc.) globally.
# If you run multiple simulations in the same script, this prevents conflicts by ensuring a fresh start.

from matplotlib.image import imread
img = (1 - imread('sano.png'))[::-1, :, 0].T

# imread('sano.png'): Reads the image into a NumPy array.

# 1 - imread(...): Inverts the grayscale values (so black becomes white and vice versa).
# [::-1, :, 0]: Flips the image vertically (so the bottom of the image corresponds to earlier time points).
# .T: Transposes the array so that rows become neurons, and columns represent time samples.

# Output: A 2D NumPy array where:
# Each column represents the stimulus at a particular time step.
# Each row represents a different neuron.

num_samples, N = img.shape

# num_samples: The number of time steps (i.e., how many columns in the array).
# N: The number of neurons (i.e., how many rows in the array).

ta = TimedArray(img, dt=1*ms)

# Defines a time-dependent input function.
# Neurons will receive input based on the intensity of the corresponding pixel at each time step.

# TimedArray(img, dt=1*ms):
# Uses the img array as a function that changes every millisecond.
# At time t, neuron i will receive input img[t, i].

A = 1.5
tau = 2*ms

# Set constants for neuron dynamics.
# A = 1.5: Scaling factor for input current.
# tau = 2*ms: Time constant, determining how quickly the neuron reacts to changes in input.

eqs = '''
dv/dt = (A*ta(t, i)-v)/tau+0.8*xi*tau**-0.5 : 1
'''

# Defines how membrane potential v evolves over time for each neuron.

# dv/dt = ...: Defines how voltage changes (v) over time.
# (A*ta(t, i) - v)/tau:
# ta(t, i): Time-varying input (from image).
# A*ta(t, i): Scaled input current.
# - v / tau: Exponential decay towards the input-driven equilibrium voltage.
# + 0.8 * xi * tau**-0.5:
# xi: Represents Gaussian white noise.
# 0.8 * xi * tau**-0.5: Introduces random fluctuations (stochasticity).

# Neuron type: A simple leaky integrator model with noise.

G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='euler')

# Defines a population of N neurons with the equation specified above.

# NeuronGroup(N, eqs, ...): Creates N neurons, each obeying the equations defined in eqs.
# threshold='v>1': A neuron spikes when its membrane potential v exceeds 1.
# reset='v=0': After a spike, the membrane potential resets to 0.
# method='euler': Uses Euler’s method to numerically integrate the equations.

M = SpikeMonitor(G)

# Records the times and neuron indices whenever a spike occurs.

run(num_samples*ms)

# Simulates for as many milliseconds as there are image columns (num_samples).

# Plot spike raster plot
plt.figure(figsize=(10, 6))
plt.plot(M.t/ms, M.i, '.k', ms=2)
plt.xlim(0, num_samples)
plt.ylim(0, N)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title("Spike Raster Plot")
plt.show()

# =========================
# Save Spiking Data to CSV
# =========================

# Each row corresponds to a neuron, with each column containing a spike time.

spike_dict = {i: [] for i in range(N)}  # Dictionary to store spike times per neuron

# Organize spike data
for neuron, time in zip(M.i, M.t):
    spike_dict[neuron].append(time/ms)  # Store spike time in milliseconds

# Convert dictionary to DataFrame
max_spikes = max(len(spike_times) for spike_times in spike_dict.values())  # Find max spikes for padding
spike_data = np.full((N, max_spikes), np.nan)  # Initialize array with NaNs (for missing values)

for i, spikes in spike_dict.items():
    spike_data[i, :len(spikes)] = spikes  # Store spike times for each neuron

df = pd.DataFrame(spike_data)

# Save to CSV file
df.to_csv("spikes.csv", index_label="Neuron Index")

# Print confirmation message
print("Spiking activity saved to 'spikes.csv'")
