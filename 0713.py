import numpy as np
import matplotlib.pyplot as plt

def amplitude_function(frequency):
    # 在这里定义您的幅度函数A(f)
    # 这里仅作为示例，输出频率的绝对值的倒数
    amplitude = 1 / np.abs(frequency)
    return amplitude
def amplitude_function_1(frequency):
    amplitude = np.zeros_like(frequency)
    
    # 在特定频率值附近设置锯齿
    width = 5000000
    
    amplitude[np.abs(frequency - 100000000) <= width] = 1
    amplitude[np.abs(frequency + 80000000) <= width] = 1
    amplitude[np.abs(frequency + 20000000) <= width] = 1
    
    return amplitude
def amplitude_function_2(frequency):
    # 定义高斯分布的参数
    mean_1 = 100000000  # 第一个高斯分布的均值
    std_dev_1 = 5000000  # 第一个高斯分布的标准差

    mean_2 = -100000000  # 第二个高斯分布的均值
    std_dev_2 = 5000000  # 第二个高斯分布的标准差
    
    # 计算高斯分布的幅度
    amplitude = np.exp(-0.5 * ((frequency - mean_1) / std_dev_1)**2)
    amplitude += np.exp(-0.5 * ((frequency - mean_2) / std_dev_2)**2)
    amplitude += np.exp(-0.5 * (frequency / std_dev_2)**2)

    return amplitude
def amplitude_function_3(frequency):
    # 定义高斯分布的参数
    mean_1 = 80
    std_dev_1 = 5
    mean_2 = -80
    std_dev_2 = 5
    mean_3 = 0
    std_dev_3 = 5
    mean_4 = -60
    std_dev_4 = 5
    
    # 计算高斯分布的幅度
    amplitude = np.exp(-0.5 * ((frequency - mean_1) / std_dev_1)**2) * 0.5
    amplitude += np.exp(-0.5 * ((frequency - mean_2) / std_dev_2)**2) * 0.5
    amplitude += np.exp(-0.5 * ((frequency - mean_3) / std_dev_3)**2) * 0.1
    amplitude += np.exp(-0.5 * ((frequency - mean_4) / std_dev_4)**2) * 0.3

    return amplitude
def phi_function(f):

    # 在此处定义相位（phi）和频率（f）之间的关系
    phi = 10*np.sin(2 * np.pi * f)  # 这是一个简单的示例，可以根据需要进行修改
    return phi
def phi_function_1(f):

    # 在此处定义相位（phi）和频率（f）之间的关系
    phi = f # 这是一个简单的示例，可以根据需要进行修改
    return phi
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
def plot_frequency_domain(A_w, phi_w):

    # 绘制频域信号的幅度和相位
    plt.figure(figsize=(12, 4))
    # 幅度图
    plt.subplot(1, 2, 1)
    plt.plot(f, A_w)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Amplitude in Frequency Domain')

    # 相位图
    plt.subplot(1, 2, 2)
    plt.plot(f, phi_w)
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.title('Phase in Frequency Domain')

    # 调整子图之间的间距
    plt.tight_layout()

    # 展示图形
    plt.show()

'''def plot_time_domain(A_t, phi_t):



    # 绘制时域信号的幅度和相位
    plt.figure(figsize=(12, 4))
    
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
    plt.show()'''
#程序开始辣！！！！！！！！！！！！！！！
#一.程序输入模块：输入随机的时域相位，目标的相位信号并展示
#1.1时间信号
# 生成一个时间轴，这个时间轴很重要！
T=0.1
t = np.linspace(0, T, 1000)
# 设置复振幅和相位
amplitude = complex(1, 0)  # 复振幅的实部和虚部
phit = np.pi*t*0  # 相位
# 生成信号
signal = amplitude * np.exp(1j*(phit))

#1.2现在开始考虑频域信号的事。

# 进行离散傅里叶变换
X = np.fft.fft(signal)

# 生成频率轴
N = len(signal)  # 信号的长度
dt = T/N  # 采样间隔
freqs = np.fft.fftfreq(N, dt)
print(freqs)

# 计算傅里叶变换结果的振幅谱和相位谱
amplitude_spectrum = np.abs(X)
phase_spectrum = np.angle(X)

# 输入你想要的傅里叶变换
w=freqs
mu = 100  # 均值
sigma = 30  # 标准差
amplitude_spectrum_0 = amplitude_function_3(w)
phase_spectrum_0 = np.zeros_like(w)

#1.3给你看看输入了啥
# 绘制振幅的实部和虚部关于时间的函数图像
plt.subplot(321)
plt.plot(t, signal.real, label='Real')
plt.plot(t, signal.imag, label='Imaginary')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Amplitude (Real and Imaginary)')
plt.legend()

# 绘制相位关于时间的图像
plt.subplot(322)
plt.plot(t, phit)
plt.xlabel('Time')
plt.ylabel('Phase')
plt.title('Phase')

# 绘制振幅谱关于频率的图像
plt.subplot(323)
plt.plot(freqs, amplitude_spectrum)
plt.xlabel('Frequency')
plt.ylabel('Amplitude Spectrum')
plt.title('Amplitude Spectrum')

# 绘制相位谱关于频率的图像
plt.subplot(324)
plt.plot(freqs, phase_spectrum)
plt.xlabel('Frequency')
plt.ylabel('Phase Spectrum')
plt.title('Phase Spectrum')

freq_range = (-100, 100)
freq_mask = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])

# 绘制感兴趣的振幅谱关于频率的图像
plt.subplot(325)
plt.plot(freqs[freq_mask], amplitude_spectrum_0[freq_mask])
plt.xlabel('Frequency')
plt.ylabel('Amplitude Spectrum')
plt.title('Amplitude Spectrum')

# 绘制振幅谱关于频率的图像
plt.subplot(321)
plt.plot(freqs, amplitude_spectrum_0)
plt.xlabel('Frequency')
plt.ylabel('Amplitude Spectrum')
plt.title('Amplitude Spectrum')

# 绘制相位谱关于频率的图像
plt.subplot(326)
plt.plot(freqs, phase_spectrum_0)
plt.xlabel('Frequency')
plt.ylabel('Phase Spectrum')
plt.title('Phase Spectrum')

# 调整子图布局
plt.tight_layout()

# 显示图像
plt.show()

#二.很好！现在你看见了我们的输入的光谱和目标光谱，现在可以很方便的进行迭代！

'''amplitude
phit
amplitude_spectrum
phase_spectrum
amplitude_spectrum_0
phase_spectrum_0'''
num_iterations = 1000
errors = []

# 执行循环
for i in range(num_iterations):
    error = calculate_mse(amplitude_spectrum, amplitude_spectrum_0)
    # 将误差添加到列表中
    errors.append(error)
    amplitude_spectrum=amplitude_spectrum_0
    A_t, phi_t = fourier_transform(amplitude_spectrum, phase_spectrum)
    #plot_time_domain(A_t, phi_t,i)
    At_in = np.mean(A_t)
    A_t=At_in* np.ones_like(A_t)
    amplitude_spectrum,  phase_spectrum=fourier_transformt(A_t, phi_t)
    #plot_frequency_domain(A_w, phi_w,i)
    i=i+1
   # print(i)
#三.很好，那我们开始画图吧
# 绘制误差随循环次数递减的图形
plt.plot(list(range(num_iterations))[10:], errors[10:])
plt.xlabel('循环次数')
plt.ylabel('误差')
plt.title('误差随循环次数变化')
plt.show()

amplitude_spectrum,  phase_spectrum
A_t, phi_t

plt.subplot(321)
plt.plot(t, A_t, label='Real')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Amplitude (Real and Imaginary)')
plt.legend()

# 绘制相位关于时间的图像
plt.subplot(322)
plt.plot(t, phi_t)
plt.xlabel('Time')
plt.ylabel('Phase')
plt.title('Phase')


# 绘制振幅谱关于频率的图像
plt.subplot(323)
plt.plot(freqs, amplitude_spectrum)
plt.xlabel('Frequency')
plt.ylabel('Amplitude Spectrum')
plt.title('Amplitude Spectrum')


# 绘制相位谱关于频率的图像
plt.subplot(324)
plt.plot(freqs, phase_spectrum)
plt.xlabel('Frequency')
plt.ylabel('Phase Spectrum')
plt.title('Phase Spectrum')

freq_range = (-1000000, 1000000)
freq_mask = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])

# 绘制感兴趣的振幅谱关于频率的图像
plt.subplot(325)
plt.plot(freqs[freq_mask], amplitude_spectrum[freq_mask])
plt.xlabel('Frequency')
plt.ylabel('Amplitude Spectrum')
plt.title('Amplitude Spectrum')
# 调整子图布局
plt.tight_layout()



# 显示图像


plt.show()
np.savetxt('phi_t.txt', phi_t)
np.savetxt('A_w.txt',amplitude_spectrum)


'''# 设置频率范围和采样点数
f_min = -400000000
f_max = 400000000
num_samples = 400000

# 计算频率轴
f = np.linspace(f_min, f_max, num_samples)

# 调用幅度函数并获取输出
amplitude = amplitude_function_3(f)
phi = phi_function_1(f)
plot_frequency_domain(amplitude, phi)
A_t, phi_t = fourier_transform(amplitude, phi)
A_w,phi_w= fourier_transformt(A_t, phi_t)
plot_frequency_domain(A_w, phi_w)

num_iterations = 20
errors = []
# 执行循环
for i in range(num_iterations):
    error = calculate_mse(A_w, amplitude)
    # 将误差添加到列表中
    errors.append(error)
    A_w=amplitude
    A_t, phi_t = fourier_transform(A_w, phi_w)
    #plot_time_domain(A_t, phi_t,i)
    At_in = np.mean(A_t)
    A_t=At_in* np.ones_like(A_t)
    A_w, phi_w=fourier_transformt(A_t, phi_t)
    #plot_frequency_domain(A_w, phi_w,i)
    i=i+1
    print("Hello, World!")

freqs = np.fft.fftfreq(len(t), t[1]-t[0])  # 计算频率轴'''



















