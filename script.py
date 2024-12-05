import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from numpy import correlate
from scipy.stats import poisson, gamma, kstest
from statistics import mean

file = "2022-01-20T15-00-00vHip1_big_filtered_sorted.mat"

all_spike_times = []
with h5py.File(file, "r") as hf:
    for channel_name in hf.keys():
        if 'times' in hf[channel_name]:
            spike_times = hf[channel_name]['times'][:]
            all_spike_times.extend(spike_times)

print(f"Liczba zebranych czasów spajków: {len(all_spike_times)}")

spike_times_dict = {}
with h5py.File(file, "r") as hf:
    for group_name in hf.keys():
        if 'times' in hf[group_name]:
            spike_times = hf[group_name]['times'][:]
            spike_times_dict[group_name] = spike_times

print(f"\nCollected spike times from {len(spike_times_dict)} groups.")

num_channels = len(spike_times_dict)
sampling_rate = 1000  
num_steps = 12600 * sampling_rate  

spike_matrix = np.zeros((num_channels, num_steps))

for i, (channel_name, spike_times) in enumerate(spike_times_dict.items()):
    spike_times = spike_times.flatten()
    for spike_time in spike_times:
        index = int(spike_time * sampling_rate)  
        if index < num_steps:
            spike_matrix[i, index] = 1

plt.figure(figsize=(100, 6))

for i in range(num_channels):
    spike_indices = np.where(spike_matrix[i, :] == 1)[0]
    plt.scatter(spike_indices / sampling_rate, np.full_like(spike_indices, i), s=2, color='purple')

plt.xlabel("Czas (s)")
plt.ylabel("Kanały")
plt.xlim([0, 60])  
plt.ylim(-0.5, num_channels - 0.5)
plt.title("Wykres spajków dla wszystkich kanałów")
plt.show()

channel = 'V2022_01_20T15_00_00vHip1_big_filtered_sorted_Ch1'
with h5py.File(file, "r") as hf:
    print(f"Zawartość kanału '{channel}': {list(hf[channel].keys())}")

    spikes = hf[channel]['times'][:] 
    print("Dane 'times':", spikes)

    num_times = spikes.size
    print(f"Liczba danych w times: {num_times}")

plt.figure(figsize=(15, 2))
plt.plot(spikes, np.ones_like(spikes), '.', color='purple')
plt.xlim([0, 1800])  
plt.ylim([0, 2])
plt.xlabel('Czas (s)')
plt.title('Wykres spajków dla kanału Ch1')
plt.yticks([1], ['Ch1'])
plt.grid(True)
plt.show()

time_range = (spikes >= 0) & (spikes <= 1800)
spikes_in_range = spikes[time_range]

num_spikes_in_range = len(spikes_in_range)
print(f"Liczba spajków w przedziale 0-1800 s: {num_spikes_in_range}")

channel10 = 'V2022_01_20T15_00_00vHip1_big_filtered_sorted_Ch10'
with h5py.File(file, "r") as hf:
    spikes10 = hf[channel10]['times'][:]

plt.figure(figsize=(50, 2))
plt.plot(spikes, np.ones_like(spikes), '.', color='purple')
plt.plot(spikes10, 2 * np.ones_like(spikes10), '.', color='red')
plt.xlim([0, 1800])
plt.ylim([0, 3])
plt.xlabel('Time (s)')
plt.yticks([1, 2], ['Ch1', 'Ch10'])
plt.show()

#ISI
ISIs = np.diff(spikes)
print(ISIs)

print("Typ danych ISIs:", type(ISIs))
print("Kształt ISIs:", np.array(ISIs).shape)

print("Min wartość ISIs:", np.min(ISIs))
print("Max wartość ISIs:", np.max(ISIs))

ISIs_flatten = ISIs.flatten()
bins = np.arange(0, 108, 0.5)
plt.figure(figsize=(10, 4))
plt.hist(ISIs_flatten, bins=bins, color='purple')
plt.xlabel('ISI [s]')
plt.ylabel('Counts')
plt.ylim([0, 30])
plt.title('Histogram ISI')
plt.show()

increments1, _ = np.histogram(spikes, bins=np.arange(0, np.max(spikes), 1))
fano_factor = increments1.var() / increments1.mean()
print('FF =', fano_factor)

N = increments1.shape[0]
shape = (N - 1) / 2
scale = 2 / (N - 1)

FF = np.linspace(0.5, 1.5, 1000)
Y = gamma.pdf(FF, shape, scale=scale)

plt.figure(figsize=(10, 6))
plt.plot(FF, Y, color='purple')
plt.xlabel('Fano Factor')
plt.ylabel('Probability density')
plt.title('Gamma distribution for FF')
plt.grid(True)
plt.show()

print(f'FF: {fano_factor}')
confidence_interval = gamma.ppf([0.025, 0.975], shape, scale=scale)
print(f'Przedział ufności dla FF: {confidence_interval}')


#autokorelacja
def autocorr(x, lags):
    xcorr = correlate(x - x.mean(), x - x.mean(), 'full') 
    xcorr = xcorr[xcorr.size//2:] / xcorr.max()
    return xcorr[:lags+1]

time_bins = np.arange(0, 30, 0.001)
increments1, _ = np.histogram(spikes, time_bins)
acf = autocorr(increments1, 100)

plt.figure(figsize=(20, 4))
plt.plot(range(len(acf)), acf, marker='o', color='purple')
plt.xlabel('Lag (ms)')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation function')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(acf, '.')
N1 = len(increments1)
sig = 2 / np.sqrt(N1)
plt.plot([0, 100], [sig, sig], 'r:')
plt.plot([0, 100], [-sig, -sig], 'r:')
plt.xlim([0, 100])
plt.ylim([-.1, .1])
plt.title('Autocorrelation function')
plt.xlabel('Time [ms]')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

# spłaszczenie tablicy, jeśli ma więcej niż jeden wymiar
if ISIs.ndim > 1:
    ISIs = ISIs.flatten()

def autocorr(x, lags):
    xcorr = correlate(x - np.mean(x), x - np.mean(x), 'full')  
    xcorr = xcorr[xcorr.size//2:] / np.max(np.abs(xcorr))  
    return xcorr[:lags+1]

ISI_acf = autocorr(ISIs, 50)

plt.figure(figsize=(10, 4))
plt.plot(ISI_acf, '.', color='purple')
N3 = len(ISIs) 
sd = 1 / np.sqrt(N3)  

plt.plot(2 * sd * np.ones_like(ISI_acf), 'r:')
plt.plot(-2 * sd * np.ones_like(ISI_acf), 'r:')
plt.xlim([0, 50])
plt.ylim([-.2, .2])
plt.xlabel('Number of spikes in the past')
plt.ylabel('Autocorrelation')
plt.show()

# isi probability hist
bins = np.arange(0, .5, 0.001)
counts, _ = np.histogram(ISIs, bins)
prob = counts / len(ISIs)
fig, ax = plt.subplots(figsize=(20, 4))
ax.stem(bins[:-1], prob)
ax.set_xlim([0, 0.15])
plt.xlabel('ISI [s]')
plt.ylabel('Probability')
plt.title('Histogram of ISI probability')
plt.show()

#ISI histogram and Poisson fit
bins = np.linspace(0, max(ISIs), 50)

plt.figure(figsize=(12, 6))

counts, bins, _ = plt.hist(ISIs, bins=bins, density=True, alpha=0.6, label="Empirical ISI")
poisson_model = poisson.pmf(np.arange(len(bins)-1), mean(ISIs))
plt.plot(bins[:-1], poisson_model[:len(bins)-1], 'r-', label="Poisson model")
plt.title("ISI histogram and Poisson fit")
plt.xlabel("ISI")
plt.ylabel("Probability density")
plt.legend()
plt.show()

# histogram przedstawia dane empiryczne, czyli rzeczywisty rozkład odstępów czasowych między zdarzeniami (ISI)
# krzywa Poissona reprezentuje teoretyczny model, który próbujemy dopasować do tych danych

plt.figure(figsize=(12, 6))
emp_cdf = np.cumsum(np.histogram(ISIs, bins=bins, density=True)[0]) * np.diff(bins)
poisson_cdf = poisson.cdf(np.arange(len(bins)-1), mean(ISIs))

# porównanie empirycznej i teoretycznej dystrybuanty
plt.plot(bins[:-1], emp_cdf, label="Empirical CDF", color='blue')
plt.plot(bins[:-1], poisson_cdf[:len(bins)-1], 'r--', label="Poisson CDF")
plt.title("CDF comparison")
plt.xlabel("ISI")
plt.ylabel("CDF")
plt.legend()
plt.show()

gamma_params = gamma.fit(ISIs)

plt.figure(figsize=(12, 6))
plt.hist(ISIs, bins=bins, density=True, alpha=0.6, label="Empirical ISI")
plt.plot(bins, gamma.pdf(bins, *gamma_params), 'g-', label="Gamma model")
plt.title("ISI Histogram with Gamma fit")
plt.xlabel("ISI")
plt.ylabel("Probability density")
plt.legend()
plt.show()


#ISI histogram vs Poisson expected distribution
x_values = np.linspace(0, max(ISIs), 1000) 
expected_isi = (1 / mean(ISIs)) * np.exp(-x_values / mean(ISIs))

plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(ISIs, bins=30, alpha=0.6, color='purple', label='Observed ISIs (histogram)', density=True)

bin_width = bins[1] - bins[0]  
probabilities = n / (len(ISIs) * bin_width)  

plt.plot(x_values, expected_isi, color='red', label='Expected ISI (Poisson)', linewidth=2)

plt.xlabel('ISI (s)', fontsize=14)
plt.ylabel('Probability density', fontsize=14)
plt.title('ISI histogram vs Poisson expected distribution', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# porównanie CDF obserwowanego ISI z teoretycznym dla rozkładu Poissona
obs_cdf = np.cumsum(np.histogram(ISIs, bins=30, density=True)[0]) 
teo_cdf = 1 - np.exp(-np.linspace(0, max(ISIs), len(obs_cdf)) / mean(ISIs)) 

# CDF
plt.figure(figsize=(10, 6))
plt.step(np.linspace(0, max(ISIs), len(obs_cdf)), obs_cdf, where='mid', label='observed CDF', color='purple')
plt.plot(np.linspace(0, max(ISIs), len(teo_cdf)), teo_cdf, label='theoretical CDF (Poisson)', color='red', linewidth=2)
plt.xlabel('ISI (s)', fontsize=14)
plt.ylabel('Cumulative probability', fontsize=14)
plt.title('Cumulative distribution function (CDF)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

#Kolmogorov - Smirnov
obs_cdf = np.cumsum(np.histogram(ISIs, bins=100, density=True)[0])  
obs_cdf /= obs_cdf[-1]

x_values = np.linspace(0, max(ISIs), len(obs_cdf))
teo_cdf = 1 - np.exp(-x_values / np.mean(ISIs))

ks_stat, p_value = kstest(ISIs, lambda x: 1 - np.exp(-x / np.mean(ISIs)))

plt.figure(figsize=(10, 6))
plt.step(x_values, obs_cdf, where='mid', label='empirical CDF', color='purple', linewidth=2)
plt.plot(x_values, teo_cdf, label='theoretical CDF', color='red', linewidth=2)

plt.xlabel('ISI (s)', fontsize=14)
plt.ylabel('Cumulative probability', fontsize=14)
plt.title('KS test - empirical vs theoretical CDF', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

#Kolmogorov - Smirnov plot
x_values = np.linspace(0, max(ISIs), 1000)
teo_cdf = 1 - np.exp(-x_values / np.mean(ISIs))  
obs_cdf = np.cumsum(np.histogram(ISIs, bins=100, density=True)[0])  
obs_cdf /= obs_cdf[-1]  

obs_cdf_resampled = np.interp(x_values, np.linspace(0, max(ISIs), len(obs_cdf)), obs_cdf)

plt.figure(figsize=(8, 8))
plt.scatter(teo_cdf, obs_cdf_resampled, color='purple', s=10, label='empirical vs theoretical CDF')
plt.plot([0, 1], [0, 1], 'r--', label='y=x')

plt.xlabel('Theoretical CDF (Poisson)', fontsize=14)
plt.ylabel('Empirical CDF', fontsize=14)
plt.title('KS plot', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()