import numpy as np
import soundfile as sf

fs = 44100
duration = 3.0
t = np.linspace(0, duration, int(fs*duration), endpoint=False)
f0, f1 = 200, 2000
signal = 0.5 * np.sin(2 * np.pi * (f0*t + (f1-f0)/(2*duration)*t**2))
sf.write('chirp.wav', signal, fs, subtype='PCM_16')