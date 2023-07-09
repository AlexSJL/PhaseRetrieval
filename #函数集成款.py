#函数集成款
import numpy as np
import matplotlib.pyplot as plt
w = np.linspace(-1000, 1000, 10000)
M = len(w)
t = np.linspace(-100, 100, M)
At_in=0.001
i=0
# 创建频率域信号
# 创建高斯振幅分布函数
mu = 50  # 均值
sigma = 10  # 标准差
A0_w = np.exp(-(w - mu)**2 / (2 * sigma**2))
phi0_w = 2 * np.pi * w 

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
plot_frequency_domain(A0_w, phi0_w,i)
A_t, phi_t = fourier_transform(A0_w, phi0_w)
plot_time_domain(A_t, phi_t,i)
A_t=At_in
A_w, phi_w=fourier_transformt(A_t, phi_t)
i=i+1
for i in range(1, 1000):

  #plot_frequency_domain(A_w, phi_w,i)
  A_w=A0_w
  A_t, phi_t = fourier_transform(A_w, phi_w)
  #plot_time_domain(A_t, phi_t,i)
  A_t=At_in
  A_w, phi_w=fourier_transformt(A_t, phi_t)
  i=i+1


plot_frequency_domain(A_w, phi_w,i)
A_w=A0_w
A_t, phi_t = fourier_transform(A_w, phi_w)
plot_time_domain(A_t, phi_t,i)







