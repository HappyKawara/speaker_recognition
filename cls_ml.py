import sounddevice as sd
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import sys
import wave
import time
import threading
import whisper
import soundfile as sf
import librosa
import scipy
from sklearn import svm
import pickle
from cnn123 import MFCC


device_list = sd.query_devices()
print(device_list)


class Recod():
    def __init__(self):
        self.save_data = []
        self.i = 0
        self.x = 0
        self.t = -21
        self.tim = 0
        self.tim_sta = 0
        self.z = 0
        self.y = 0
        self.num = 0
        self.save_ls = np.array([0])
        sd.default.device = [7, 7] # Input, Outputデバイス指定



    def callback(self,indata,frames,ti,status):

        # indata.shape=(n_samples, n_channels)
        
        data = indata[::self.downsample, 0]
        self.save_data =  np.append(self.save_data,data)
        shift = len(data)
        self.plotdata = np.roll(self.plotdata, -shift, axis=0)
        self.plotdata[-shift:] = data
        self.time_end = time.time()
        if (self.time_end - self.time_sta) > 0.2 * (self.i+1):
            avg = np.mean(np.abs(self.plotdata[:-1000]))
            if avg < 0.01:
                self.i = self.i + 1
                print(self.i,avg,data.shape)
                if self.y > 3:
                    self.z = self.z + 1
                else:
                    self.save_data = np.array([])
            else:
                self.y = self.y + 1
                print("listen",avg)
                self.i += 1
            if self.z > 3:
                self.z = 0
                self.y = 0
                self.save_data = self.save_data / self.save_data.max() * np.iinfo(np.int16).max
                self.save_data = self.save_data.astype(np.int16)
                with wave.open('./wav_file/test{}.wav'.format(str(self.num)), mode='w') as wb:
                    wb.setnchannels(1)  # モノラル
                    wb.setsampwidth(2)  # 16bit=2byte
                    wb.setframerate(44100)
                    wb.writeframes(self.save_data.tobytes())  # バイト列に変換
                wav, fs = sf.read('./wav_file/test{}.wav'.format(str(self.num)))
                wav_trimmed, index = librosa.effects.trim(wav, top_db=25)
                wav_filtered = scipy.signal.wiener(wav_trimmed)
                sf.write(file='./wav_file/test{}.wav'.format(str(self.num + 1000)),data=wav_filtered,samplerate=fs)
                self.save_data = []
                self.z = 0
                self.y = 0
                thread = threading.Thread(target=self.func,args=(self.num,))
                thread.start()
                self.num += 1
        #if data[-1] <
        #print(self.save_data.shape)

    def func(self,num):
        y = MFCC()
        x = y.cnn_predict('./wav_file/test{}.wav'.format(str(num+1000)))
        print(x)
        """
        x, fs = sf.read('./wav_file/test{}.wav'.format(str(num+1000)))#BASIC5000_0641.wav')#test1.wav')
        mfccs = librosa.feature.mfcc(y = x, sr=fs,n_mfcc=12,dct_type=3)
        with open('sample_test.pkl','rb') as f:
            classifier = pickle.load(f)
        data = classifier.predict(mfccs.T)
        print(data)
        ls = [0,0,0,0]
        for d in data:
            ls[d] += 1
        if ls[1] > ((ls[2] + ls[3]) or 5):
            print(1)
        if ls[2] > ((ls[1] + ls[3]) or 5):
            print(2)
        elif ls[3] > ((ls[1] + ls[2]) or 5):
            print(3)
        else:
            print(False)

        x, fs = sf.read('./wav_file/test{}.wav'.format(str(num)))#BASIC5000_0641.wav')#test1.wav')
        mfccs = librosa.feature.mfcc(y = x, sr=fs,n_mfcc=12,dct_type=3)
        """
        """
        with open('sample_test_6.pkl','rb') as f:
            classifier = pickle.load(f)
        data = classifier.predict(mfccs.T)
        print(data)
        ls = [0,0,0,0,0,0,0,0,0]
        for d in data:
            ls[d] += 1
        if ls[1] > ((ls[2] + ls[3]) or 5):
            print(1)
        elif ls[2] > ((ls[1] + ls[3]) or 5):
            print(2)
        elif ls[3] > ((ls[1] + ls[2]) or 5):
            print(3)
        else:
            print(False)
        """

    '''
            i = 0
            self.time_sta = time.time()
        if i == 500:
            # 時間計測終了
            self.time_end = time.time()
            # 経過時間（秒）
            tim = self.time_end - self.time_sta
            print(tim)
            sys.exit()
    '''
    def update_plot(self,frame):
        """This is called by matplotlib for each plot update.
        """
        self.plotdata
        self.line.set_ydata(self.plotdata)
        return self.line,

    def main(self):
        self.downsample = 1
        length = int(1000 * 44100 / (1000 * self.downsample))
        self.plotdata = np.zeros((length))

        fig, ax = plt.subplots()
        self.line, = ax.plot(self.plotdata)
        ax.set_ylim([-1.0, 1.0])
        ax.set_xlim([0, length])
        ax.yaxis.grid(True)
        fig.tight_layout()
        self.time_sta = time.time()
        with sd.InputStream(
                channels=1,
                dtype='float32',
                callback=self.callback
                ):
                sd.sleep(int(1000 * 1000))



#ani = FuncAnimation(fig, update_plot, interval=30, blit=True)

#with stream:
            #plt.show()
if __name__ == '__main__':
    Recod().main()
    print("a")
