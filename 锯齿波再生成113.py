#函数集成款


import numpy as np
import matplotlib.pyplot as plt

def sawtooth_wave(t, T):
    return 2*np.pi * (t / T - np.floor(0.5 + t / T))
''''''
# 设置时间范围和采样点数
t = np.linspace(0, 100, 50000)

# 创建锯齿波信号
T = 3  # 锯齿波的周期
sawtooth_phase = sawtooth_wave(t, T)
amplitude = 1  # 振幅
sawtooth_signal = amplitude * np.exp(1j * sawtooth_phase)

# 进行傅里叶变换
frequency_domain = np.fft.fft(sawtooth_signal)
freqs = np.fft.fftfreq(len(t), t[1]-t[0])  # 计算频率轴



import numpy as np
import matplotlib.pyplot as plt
w = np.linspace(-1000, 1000, 100000)
M = len(w)
t = np.linspace(0, 100, M)
At_in=1
i=0
w=freqs
# 创建频率域信号
# 创建高斯振幅分布函数
mu = 10  # 均值
sigma = 1  # 标准差
#A0_w=np.exp(-(w - mu)**2 / (2 * sigma**2))
'''
A0_w = np.zeros_like(w)
index = np.abs(w - 1).argmin()
A0_w[index] = 1'''

'''A0_w = np.zeros_like(w)
A0_w[(w > 18) & (w < 22)] = 1
A0_w[(w > 38) & (w < 42)] = 1
A0_w[(w > 58) & (w < 62)] = 1
phi0_w =phase_spectrum = np.zeros_like(frequency_domain)'''
A0_w = amplitude_spectrum = np.abs(frequency_domain)
phi0_w =phase_spectrum = np.angle(frequency_domain)


def plot_frequency_domain(A_w, phi_w,i):

    # 绘制频域信号的幅度和相位
    plt.figure(figsize=(12, 4))
    # 标题
    plt.suptitle(f"Iteration: {i}")
    # 幅度图
    plt.subplot(1, 2, 1)
    plt.plot(w, A_w)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Amplitude in Frequency Domain')

    # 相位图
    plt.subplot(1, 2, 2)
    plt.plot(w, phi_w)
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.title('Phase in Frequency Domain')

    # 调整子图之间的间距
    plt.tight_layout()

    # 展示图形
    plt.show()

def plot_time_domain(A_t, phi_t,i):



    # 绘制时域信号的幅度和相位
    plt.figure(figsize=(12, 4))
    # 标题
    plt.suptitle(f"Iteration: {i}")
    # 幅度图
    plt.subplot(1, 2, 1)
    plt.plot(t, A_t)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Amplitude in Time Domain')

    # 相位图
    plt.subplot(1, 2, 2)
    plt.plot(t, phi_t)
    plt.xlabel('Time')
    plt.ylabel('Phase')
    plt.title('Phase in Time Domain')

    # 调整子图之间的间距
    plt.tight_layout()

    # 展示图形
    plt.show()

def fourier_transform(A_w, phi_w):
    
    # 构建频域信号
    signal = A_w * np.exp(1j * phi_w)
    
    # 傅里叶变换到时域
    time_signal = np.fft.ifft(signal)
    
    # 提取时域信号的幅度和相位
    A_t = np.abs(time_signal)

    phi_t = np.angle(time_signal)
    
    return A_t, phi_t

def fourier_transformt(A_t, phi_t):
    
    # 构建频域信号
    signal = A_t * np.exp(1j * phi_t)
    
    # 傅里叶变换到时域
    time_signal = np.fft.fft(signal)
    
    # 提取时域信号的幅度和相位
    A_w = np.abs(time_signal)

    phi_w = np.angle(time_signal)
    
    return A_w, phi_w

def calculate_mse(signal1, signal2):
    # 确保信号长度相同
    if len(signal1) != len(signal2):
        raise ValueError("信号长度不一致")

    # 计算均方误差
    mse = np.mean((signal1 - signal2)**2)

    return mse


#频率上
plot_frequency_domain(A0_w, phi0_w,i)
A_t, phi_t = fourier_transform(A0_w, phi0_w)
#np.savetxt('A1_w.txt', phi_t)
#plot_time_domain(A_t, phi_t,i)
At_in = np.mean(A_t)
'''A_t=At_in*np.ones_like(A_t)
phi_t = 2 * np.pi * np.ones_like(A_t)*t
plot_time_domain(A_t, phi_t,10000)'''

A_w, phi_w=fourier_transformt(A_t, phi_t)

i=i+1

'''
for i in range(1, 1000):

  #plot_frequency_domain(A_w, phi_w,i)
  A_w=A0_w
  A_t, phi_t = fourier_transform(A_w, phi_w)
  #plot_time_domain(A_t, phi_t,i)
  A_t=At_in
  A_w, phi_w=fourier_transformt(A_t, phi_t)
  i=i+1
'''
# 定义循环次数和误差列表
num_iterations = 100
errors = []

# 执行循环
for i in range(num_iterations):
    error = calculate_mse(A_w, A0_w)
    # 将误差添加到列表中
    errors.append(error)
    A_w=A0_w
    A_t, phi_t = fourier_transform(A_w, phi_w)
    #plot_time_domain(A_t, phi_t,i)
    At_in = np.mean(A_t)
    A_t=At_in* np.ones_like(A_t)
    A_w, phi_w=fourier_transformt(A_t, phi_t)
    #plot_frequency_domain(A_w, phi_w,i)
    i=i+1
    print("Hello, World!")
    
# 绘制误差随循环次数递减的图形
plt.plot(range(num_iterations), errors)
plt.xlabel('循环次数')
plt.ylabel('误差')
plt.title('误差随循环次数变化')
plt.show()

plot_frequency_domain(A_w, phi_w,i)


A_w=A0_w


A_t, phi_t = fourier_transform(A_w, phi_w)
phi_t_unwrapped = np.unwrap(phi_t)
#时间相位
plot_time_domain(A_t,phi_t,i)
print(At_in)  # 打印输出常数
np.savetxt('phi_t.txt', phi_t)






