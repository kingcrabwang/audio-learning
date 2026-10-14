import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
#1、参数
sr=44100
duration=1.0
freq=1000
amplitude=0.5
#2、时间轴
t=np.linspace(0,duration,int(sr*duration),endpoint=False)
#3、生成正弦波
data=amplitude*np.sin(2*np.pi*freq*t)
#4、写入文件
sf.write('sine_1K.wav',data,sr,subtype='PCM_16')
print("已生成sine_1k.WAV")

plt.figure(figsize=(12,4))
plt.plot(t,data)

plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.title('Waveform')#设置标签和标题
plt.grid(True)
plt.tight_layout()
plt.show()
