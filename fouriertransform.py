import numpy as np
import matplotlib.pyplot as plt

# 定义时间范围和采样点数
t = np.linspace(0, 100, 10000000)
N = len(t)

# 定义原始信号的振幅和相位
A0 = 0.5
phi_j = 2 * np.pi * t+0.5+np.pi * t*t


# 在时间域中构建原始信号
signal = A0 * np.exp(1j * phi_j)+ A0 * np.exp(2j * phi_j)

# 进行傅里叶变换
freq_signal = np.fft.fft(signal)

# 计算频率轴
freq = np.fft.fftfreq(N)

# 绘制原始信号与频谱
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(t, np.real(signal), label='Real part')
ax1.plot(t, np.imag(signal), label='Imaginary part')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.legend()

ax2.plot(freq, np.abs(freq_signal))
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Amplitude Spectrum')

plt.tight_layout()
plt.show()
print("Hello, World!")