import numpy as np
import nfft
import matplotlib.pyplot as plt

### Artificially constructed data

N = 1024 * 16
x = -0.5 + np.random.rand(N)
f1 = np.sin(10 * 2 * np.pi * x) + np.cos(7 * 2 * np.pi * x) + 0.1 * np.random.randn(N) + x
f2 = np.exp(x)
f = f1

k = - N//2 + np.arange(N)

n = 1024
f_k = nfft.nfft_adjoint(x, f, n, truncated=False)

f_r = np.flip(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_k))) / (N / n))
x_r = np.linspace(-0.5, 0.5, n, endpoint=False)

plt.scatter(x[:n], f[:n], label='f')
plt.scatter(x_r, f_r, label='reconstruction')
plt.xlabel('x')
plt.ylabel('signal')
plt.legend()
plt.show()

### Our data

# use the first 30 z levels
N = 30
z = [   40.  ,   127.66,   242.99,   394.27,   591.33,   844.45,
        1161.79,  1545.32,  1987.97,  2475.32,  2991.54,  3524.26,
        4065.69,  4611.51,  5159.48,  5708.51,  6258.04,  6807.81,
        7357.7 ,  7907.65,  8457.62,  9007.61,  9557.61, 10107.6 ,
       10657.6 , 11207.6 , 11757.6 , 12307.6 , 12857.61, 13407.63,
       13957.68, 14507.77, 15057.96, 15608.36, 16159.19, 16710.92,
       17264.52, 17822.02, 18387.69, 18970.46, 19589.34, 20284.86,
       21140.86, 22292.35, 23775.74, 25399.22, 27046.11, 28695.77][:N]

# interpolate to the required domain for nfft, [-0.5, 0.5]
x = np.interp(z, (min(z), max(z)), (-0.5, +0.5))

f = train_outputs[0][0:30].numpy()

fcs_list = [4, 16, 32, 512]
cmap = plt.get_cmap('viridis', len(fcs_list))
for fcs, c in zip(reversed(fcs_list), reversed(cmap.colors)):
    f_k = nfft.ndft_adjoint(x, f, fcs)
    f_recon = np.flip(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_k))) / (N / fcs))
    x_recon = np.linspace(-0.5, +0.5, fcs, endpoint=True)
    plt.plot(x_recon, f_recon, c=c, ls='--', marker='o',label='{} FC'.format(fcs))
plt.xlabel('z (m)')
plt.ylabel('DQP (kg/kg)')
plt.plot(x, f, marker='o', label='original')
plt.legend()
