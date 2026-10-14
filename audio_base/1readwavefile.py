import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
#1、读取文件
data,sr=sf.read('chirp.wav')
#2、生成时间轴单位s
duration=len(data)/sr
time=np.arange(len(data))/sr
#3、输出各项数据
print(f"采样率：{sr}Hz")
print(f"持续时间：{duration}s")
print(f"数据形状：{data.shape}")
print(f"数据类型：{data.dtype}")
#4、绘制波形图
plt.figure(figsize=(12,4))#画布大小
plt.plot(time,data)#绘制二维线图

plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.title('Waveform')#设置标签和标题
plt.grid(True)
plt.tight_layout()
plt.show()
