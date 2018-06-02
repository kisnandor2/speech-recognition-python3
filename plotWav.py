import matplotlib.pyplot as plt
import numpy as np
import wave

spf = wave.open('audio/apple/apple01.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')

fs = spf.getframerate()

print(fs, len(signal))

Time=np.linspace(0, len(signal)/fs, num=len(signal))

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(Time, signal)
plt.show()