import sounddevice as sd
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import sys
import wave
import time
import threading
import whisper
import librosa
import soundfile as sf
from sklearn import svm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import sklearn
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
#%matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, UpSampling1D
from tensorflow.keras.layers import MaxPooling1D,MaxPooling2D
import pandas as pd
import collections
from sklearn import preprocessing
import  tensorflow as tf
import random
import itertools

np.set_printoptions(suppress=True)
STOP = 200
random.seed(0)
pkl_name = "EnvNet"

class MFCC():
    def __init__(self):
        self.fs = 48000
        self.DATA_SIZE = 6
        self.N = 48000
        self.n_mels = 64
        self.SPLICE_SIZE = int(self.n_mels)
        self.SPLICE_LANGE = int(self.SPLICE_SIZE / 4)
        self.num = 1
        self.ls = np.array([1])
        self.logmel_lists = []
        self.train_sizes = np.arange(1, 3000, 20)
        self.delta_feature_lists = []
        self.delta_delta_feature_lists = []
        self.FILESUM = 50
        self.split_len = 0
        self.bc_label = []
        self.bc_mel = []
        self.data_make_bc = []
        self.train_label_num = []
        self.fft_len = 512
        self.mel_len = 282
        for i in range(self.DATA_SIZE):
            self.data_make_bc.append([])
            self.train_label_num.append(0)

    def Machine_Learning(self):
        """
        file_path = 'wave_file/0ftcf-trlsz.wav'
        self.load_wav2(file_path,1,0)
        file_path = 'wave_file/gxszg-1uk3l.wav'
        self.load_wav2(file_path,2,0)
        #print(len(self.logmel_lists))
        """
        """
        file_path = 'wave_file/summer_pockets/ao_2.wav'
        self.load_wav2(file_path,7,0)
        #print(len(self.logmel_lists))
        file_path = 'wave_file/summer_pockets/kamome_1.wav'
        self.load_wav2(file_path,8,0)
        #print(len(self.logmel_lists))
        file_path = 'wave_file/summer_pockets/shiki_1.wav'
        self.load_wav2(file_path,9,0)
        #print(len(self.logmel_lists))
        file_path = 'wave_file/summer_pockets/shiroha_2.wav'
        self.load_wav2(file_path,10,0)
        #print(len(self.logmel_lists))
        file_path = 'wave_file/summer_pockets/tumugi_2.wav'
        self.load_wav2(file_path,11,0)
        #print(len(self.logmel_lists))
        """
        for i in range(self.FILESUM):

            file_path = 'jsut_ver1_1/basic5000/wav/BASIC5000_{}.wav'.format(str(self.num).zfill(4))
            self.load_wav2(file_path,1,i)

            file_path = 'wav2/BASIC5000_{}.wav'.format(str(self.num).zfill(4))
            self.load_wav2(file_path,2,i)

            file_path = 'wav3/BASIC5000_{}.wav'.format(str(self.num + 600).zfill(4))
            self.load_wav2(file_path,3,i)
            file_path = 'wav456/fujitou_normal/fujitou_normal_{}.wav'.format(str(self.num).zfill(3))
            self.load_wav2(file_path,4,i)
            file_path = 'wav456/tsuchiya_normal/tsuchiya_normal_{}.wav'.format(str(self.num).zfill(3))
            self.load_wav2(file_path,5,i)
            file_path = 'wav456/uemura_normal/uemura_normal_{}.wav'.format(str(self.num).zfill(3))
            self.load_wav2(file_path,6,i)
            self.num = self.num + 1
            #print(len(self.logmel_lists))
        
        print(len(self.logmel_lists),np.array(self.logmel_lists).shape)
        print(self.train_label_num)
        #self.make_bc_mel()
        self.make_mixup_mel()
        self.fitdata()



    def load_wav2(self,file_name,data_num,num):
        #最後のファイルだけテストデータにするためにリストの長さを変数に保存

        ls = []
        log_mel_specs = []

        #音声ファイルのロード
        try:
            x, fs = sf.read('./'+ file_name)
        except Exception as e:
            print(e)
            return None
        if fs != 48000:
            print("make4800fs")
            wav, _ = librosa.load('./' + file_name, sr=fs)
            resampled_wav = librosa.resample(wav, orig_sr = fs, target_sr = 48000)
            sf.write('./'+ file_name, resampled_wav, 48000, 'PCM_16')
            x, fs = sf.read('./'+ file_name)

        #print(fs)
        #print(len(x))
        #de-tamizumashisuru
        m = len(x)//(self.fs*3)
        #print(m)
        i = 0
        mels = np.reshape(x[:m*self.fs*3],(-1,self.fs*3))
        for mel in mels:
            mel = mel.tolist()
            self.logmel_lists = self.logmel_lists + mel
            self.ls = np.append(self.ls,np.full(1,data_num-1))
            self.data_make_bc[data_num-1] += mel
            self.train_label_num[data_num-1] = self.train_label_num[data_num-1] + 1

        """
        while 1:
            mels = x[i*fs*3:(i+1)*fs*3]


            #print(mels.shape)
            mel = mels.tolist()
            #mel2 = mels2.tolist()
            #mel3 = mels3.tolist()
            #print(mels.shape)
            #print(mel[0][32])

            self.logmel_lists = self.logmel_lists + mel# + mel2 + mel3
            #n,data_sum = (self.n_mels,self.fs)
            self.ls = np.append(self.ls,np.full(1,data_num-1))
            self.data_make_bc[data_num-1] += mel
            self.train_label_num[data_num-1] = self.train_label_num[data_num-1] + 1
            i += 1
            #print(i)
            if i >= STOP:
                break
            #print(len(self.ls))
        """

    def load_wav(self,file_name,data_num,num):

        ls = []
        log_mel_specs = []

        #音声ファイルのロード
        try:
            x, fs = sf.read('./'+ file_name)
        except Exception as e:
            print(e)
            return None
        #print(fs)
        #print(len(x))
        mels = librosa.feature.melspectrogram(y=x[:fs*3], sr=fs,n_mels = self.n_mels).T
        mels = librosa.power_to_db(mels,ref = np.max)
        #print(mels.shape)
        mel_len , n_mels = (mels.shape)
        if mel_len != 282:
            print(fs,mel_len)
            print("too short file")
            return None
        mel = mels.tolist()

        self.logmel_lists = self.logmel_lists + mel
        n,data_sum = (self.n_mels,mel_len)
        self.ls = np.append(self.ls,np.full(1,data_num-1))
        #print(len(self.ls))

    def make_bc_mel(self):

        m = min(self.train_label_num)
        min_index = self.train_label_num.index(m)
        random_list = []
        print(np.array(self.data_make_bc[0]).shape,m)
        for i in self.train_label_num:
            #print(i)
            random_list.append(random.sample(range(0,i-1),m-2))

        #ランダムにデータを組み合わせて水増しする
        for (ind,ls),i in itertools.product(enumerate(self.data_make_bc),range(int(m-2))):
            #ls > (-1,282,64)

            mel1 = np.array(ls[random_list[ind][i]*self.mel_len:(1+random_list[ind][i])*self.mel_len])
            #print(mel1.shape,random_list[ind][i])
            r = random.randint(0,self.DATA_SIZE-2)
            if r >= ind:
                r += 1
            mel2 = np.array(self.data_make_bc[r][random_list[r][i]*self.mel_len:
                                                 (1+random_list[r][i])*self.mel_len])
            label1 = ind
            label2 = r
            bc_mel,bc_label = self.bclearning(label1,label2,mel1,mel2)
            self.bc_mel.append(bc_mel)
            self.bc_label.append(bc_label)

    def make_mixup_mel(self):

        m = min(self.train_label_num)
        min_index = self.train_label_num.index(m)
        random_list = []
        print(np.array(self.data_make_bc[0]).shape,m)
        for i in self.train_label_num:
            #print(i)
            random_list.append(random.sample(range(0,i-1),m-2))

        #ランダムにデータを組み合わせて水増しする
        for (ind,ls),i in itertools.product(enumerate(self.data_make_bc),range(10)):#int(m-2))):
            #ls > (-1,282,64)

            mel1 = np.array(ls[random_list[ind][i]*self.mel_len:
                               (1+random_list[ind][i])*self.mel_len])
            #print(mel1.shape,random_list[ind][i])
            r = random.randint(0,self.DATA_SIZE-2)
            if r >= ind:
                r += 1
            mel2 = np.array(self.data_make_bc[r][random_list[r][i]*self.mel_len:
                                                 (1+random_list[r][i])*self.mel_len])
            label1 = ind
            label2 = r
            bc_mel,bc_label = self.mixup(label1,label2,mel1,mel2)
            self.bc_mel.append(bc_mel)
            self.bc_label.append(bc_label)
            #print(len(bc_label))


    def bclearning(self,label1,label2,mel1,mel2):
        #one-hot化
        label1_onehot = np.array(to_categorical(label1, self.DATA_SIZE))
        label2_onehot = np.array(to_categorical(label2, self.DATA_SIZE))
        #bc_learning
        r = random.random()
        g1 = np.amax(mel1)
        g2 = np.amax(mel2)
        p = 1/(1+10**((g1-g2)/20)*((1-r)/r))
        bc_label = (r * label1_onehot) +((1-r) * label2_onehot)
        #print(p,mel1.shape,mel2.shape)
        #bc_mel = ((p*mel1)+((1-p)*mel2))/((p**2+(1-p)**2)**(1/2))
        bc_mel = ((p*mel1)+((1-p)*mel2))/((p**2+(1-p)**2)**(1/2))
        #print(bc_label,r)

        return bc_mel,bc_label


    def mixup(self,label1,label2,mel1,mel2):
        #one-hot化
        label1_onehot = to_categorical(label1, self.DATA_SIZE)
        label2_onehot = to_categorical(label2, self.DATA_SIZE)
        #bc_learning
        r = random.random()
        #g1 = np.average(mel1)
        #g2 = np.average(mel2)
        #p = 1/(1+10**(g1 - g2)*((1-r)/r))
        mixup_label = (r * label1_onehot) +((1-r) * label2_onehot)
        mixup_mel = (r * mel1) +((1-r) * mel2)
        return mixup_mel,mixup_label

    def fitdata(self):

        #正規化
        mm = preprocessing.MinMaxScaler()
        self.logmel_lists = mm.fit_transform(np.reshape(self.logmel_lists,(-1,1)))

        #ニューラルネットワークで扱える形に変換
        feature = self.logmel_lists
        test_input_shape = [ -1, self.fs,1]#*3
        feature3 = np.reshape(feature, test_input_shape)
        data_num = (feature3.shape)
        print(data_num)
        self.ls = self.ls[1::]


        feature3 = np.array(feature3)
        train_labels = to_categorical(self.ls, self.DATA_SIZE)
        print("feature3:{}".format(feature3.shape))
        print("train_labels:{}".format(train_labels.shape))

        training_accuracy = []
        test_accuracy = []
        print("fit")
        print(np.array(self.bc_mel).shape,self.bc_label[:5])

        #学習用と検証用に分ける
        self.x_train = np.append(feature3[::4] , feature3[1::4],axis = 0)
        self.x_train = np.append(self.x_train, feature3[3::4],axis = 0)
        self.x_train = np.append(self.x_train , np.reshape(self.bc_mel,test_input_shape),axis = 0)
        self.x_train = np.reshape(np.array(np.array(self.x_train)),test_input_shape)
        #self.x_train = np.append(self.x_train ,self.bc_mel ,axis=0)
        self.x_test = feature3[3::4]
        self.label_train = np.append(train_labels[::4] , train_labels[1::4] , axis = 0)
        self.label_train = np.append(self.label_train , train_labels[2::4] , axis = 0)
        print("bc_label shape:{}".format(np.array(self.bc_label).shape))
        self.label_train = np.append(self.label_train, self.bc_label , axis = 0)
        self.label_test = train_labels[3::4]
        print("x_train:{}".format(self.x_train.shape))
        print("x_test:{}".format(self.x_test.shape))
        print("label_train:{}".format(self.label_train.shape))
        print("label_test:{}".format(self.label_test.shape))
        print(self.label_test)#[::10])

        #NPデータを保存
        a = (feature3,train_labels)
        with open(pkl_name +'_{}.pkl'.format(str(self.N)),'wb') as f:
            pickle.dump(a,f)

        #CNNモデル
        model = Sequential()
        model.add(Conv1D(self.SPLICE_SIZE,8, padding='same',
                         input_shape = data_num[1:],#(self.SPLICE_SIZE,self.n_mels*3),
                         activation='relu'))
        model.add(Conv1D(self.SPLICE_SIZE,8, padding='same', activation='relu'))
        model.add(MaxPooling1D(320, padding='same'))
        model.add(Dropout(0.25))
        """
        model.add(Conv2D(self.SPLICE_SIZE/2,(3,3), padding='same', activation='relu'))
        model.add(Conv2D(self.SPLICE_SIZE/2,(3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2,2), padding='same'))
        model.add(Dropout(0.25))
        """
        model.add(Conv1D(int(self.SPLICE_SIZE / 4), (3,3), padding='same', activation='relu'))

        model.add(Flatten())
        model.add(Dense(self.mel_len*2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.mel_len/4, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.DATA_SIZE, activation='softmax'))

        model.compile(loss='categorical_crossentropy',#tf.keras.losses.KLDivergence(),bc_learning
                                                      #'categorical_crossentropy',  mixup
                      optimizer=Adam(learning_rate=0.0001),
                      metrics=['acc'])
        print(model.summary())
        print(np.array(self.x_train).shape,np.array(self.label_train).shape,
              np.array(self.x_train)[0][0][0])

        history = model.fit(np.array(self.x_train).astype(np.float32),
                            np.array(self.label_train).astype(np.float32),
                            batch_size=64,epochs=32, validation_split=0.3)

        #正答率
        plt.plot(history.history['acc'], label='acc')
        plt.plot(history.history['val_acc'], label='val_acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='best')
        plt.ylim([0.0, 1.05])
        plt.show()

        #損失関数
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='best')
        #plt.ylim([0.0, 1.05])
        plt.show()

        #テストデータの正答率と損失
        test_loss, test_acc = model.evaluate(self.x_test, self.label_test)
        print('loss: {:.3f}\nacc: {:.3f}'.format(test_loss, test_acc ))

        print(self.x_test.shape)
        x = model.predict(self.x_test)
        #print(x)
        #print(self.label_test)

        #CNNモデルの保存
        with open(pkl_name +'.pkl','wb') as f:
            pickle.dump(model,f)
        print('finish')

    def cnn_predict(self,file_name):#予測
        with open(pkl_name +'.pkl','rb') as f:
            model = pickle.load(f)
        try:
            x, fs = sf.read('./'+ file_name)
        except Exception as e:
            print(e)
            return None
        if fs != 48000:
            wav, _ = librosa.load('./' + file_name, sr=fs)
            resampled_wav = librosa.resample(wav, orig_sr = fs, target_sr = 48000)
            sf.write('./'+ file_name, resampled_wav, 48000, 'PCM_16')
            x, fs = sf.read('./'+ file_name)
        mel_list  = []
        if len(x[:fs].T) != 48000:
            print("sutereo")
            i = 0
            while 1:
                mels = librosa.feature.melspectrogram(y=x[i*fs*3:(i+1)*fs*3].T[0],
                                                      sr=fs,n_mels = self.n_mels).T
                #print(mels.shape)
                mel_len , n_mels = (mels.shape)
                if mel_len != self.mel_len:
                    print("finish")
                    break
                mel = mels.tolist()
                mel_list = mel_list + mel
                i += 1
                print(i)
                if i >= STOP:
                    break

            print(mels.shape)
        else:
            print("mono")
            i = 0
            while 1:
                mels = librosa.feature.melspectrogram(y=x[i*fs*3:(i+1)*fs*3],
                                                      sr=fs,n_mels = self.n_mels,
                                                  hop_length = self.fft_len,
                                                  win_length=self.fft_len).T
                #print(mels.shape)
                mel_len , n_mels = (mels.shape)
                if mel_len != self.mel_len:
                    print("finish")
                    break
                mel = mels.tolist()
                mel_list = mel_list + mel
                i += 1
                m = i
                #print(i)a
                if i >= STOP:
                    break



        #ニューラルネットワークで扱える形に変換
        feature3 = []
        feature = mel_list
        test_input_shape = [ -1,282 ,self.n_mels,1]
        feature3 = np.reshape(feature, test_input_shape)
        data_num = (feature3.shape)


        x = model.predict(feature3)
        ls = [ np.argmax(i) for i in x]
        c = collections.Counter(ls)
        b = x

        #print(b)
        #print(c,d)
        return (c.most_common()[0],i)#,d.most_common()[0])

    def delta_parameters(self,ls):#動的特徴量
        delta_logmel_lists = []
        for i,log_mel_spec in enumerate(ls):
            delta_logmel_list = []
            for count_n_logmel,m in enumerate(log_mel_spec):
                if i != 0:
                    delta_logmel = m - ls[i-1][count_n_logmel]
                else:
                    delta_logmel = m-m
                delta_logmel_list.append(delta_logmel)
            delta_logmel_lists.append(delta_logmel_list)
        return delta_logmel_lists



if __name__ == '__main__':
    x = MFCC()
    x.Machine_Learning()
    #print(x.cnn_predict("BASIC5000_0085.wav"))
    #print(x.cnn_predict("BASIC5000_0650.wav"))
    #print(x.cnn_predict("wave_file/a.wav"))
    print("add5{}".format(x.cnn_predict("wave_file/test_add1.wav")))
    print("test5{}".format(x.cnn_predict("wave_file/test2.wav")))
    print("ka{}".format(x.cnn_predict('wave_file/0ftcf-trlsz.wav')))
    print("wai{}".format(x.cnn_predict('wave_file/gxszg-1uk3l.wav')))
    print("ao{}".format(x.cnn_predict('wave_file/summer_pockets/ao_1.wav')))
    print("kamome{}".format(x.cnn_predict('wave_file/summer_pockets/kamome_2.wav')))
    print("shiki{}".format(x.cnn_predict('wave_file/summer_pockets/shiki_3.wav')))
    print("tumugi{}".format(x.cnn_predict('wave_file/summer_pockets/tumugi_1.wav')))
    #print(x.cnn_predict("BASIC5000_0085.wav"),"label:1")
    print(x.cnn_predict("BASIC5000_0650.wav"),"label:2")
    #print(x.cnn_predict("wave_file/a.wav"),)
    #print("add5{}".format(x.cnn_predict("wave_file/test_add1.wav")))
    #print("test5{}".format(x.cnn_predict("wave_file/test2.wav")))
    print("wai{}".format(x.cnn_predict('wave_file/0ftcf-trlsz.wav')))
    print("kawara{}".format(x.cnn_predict('wave_file/gxszg-1uk3l.wav')))
    print(x.cnn_predict("wave_file/BASIC5000_0640_3.wav"),"label:2")
    print(x.cnn_predict("wave_file/fujitou_normal_096_4.wav"),"label:3")
    print("ao{}".format(x.cnn_predict('wave_file/summer_pockets/ao_2.wav')))
    print("kamome{}".format(x.cnn_predict('wave_file/summer_pockets/kamome_1.wav')))
    print("shiki{}".format(x.cnn_predict('wave_file/summer_pockets/shiki_2.wav')))
    print("tumugi{}".format(x.cnn_predict('wave_file/summer_pockets/tumugi_1.wav')))
    print(x.cnn_predict("wave_file/tsuchiya_normal_095_5.wav"),"label:4")
    print(x.cnn_predict("wave_file/uemura_normal_100_6.wav"),"label:5")
    print(x.cnn_predict("wave_file/tsuchiya_normal_099_5.wav"),"label:4")
    print(x.cnn_predict("wave_file/uemura_normal_099_6.wav"),"label:5")









