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
from tensorflow.keras.layers import MaxPooling1D
import pandas as pd
import collections

np.set_printoptions(suppress=True)

class MFCC():
    def __init__(self):
        self.DATA_SIZE = 6
        self.N = 44100
        self.n_mels = 64
        self.SPLICE_SIZE = int(self.n_mels) * 3#int(self.N/2)
        self.SPLICE_LANGE = int(self.SPLICE_SIZE / 8)
        self.num = 1
        self.ls = np.array([1])
        self.logmel_lists = []
        self.train_sizes = np.arange(1, 3000, 20)
        self.delta_feature_lists = []#np.arange(self.N_MFCC).reshape((1,self.N_MFCC))
        self.delta_delta_feature_lists = []#np.arange(self.N_MFCC).reshape((1,self.N_MFCC))
        self.length_ls = [0]
        self.FILESUM = 10
        self.split_len = 0

    def Machine_Learning(self):
        for i in range(self.FILESUM):

            file_path = 'jsut_ver1_1/basic5000/wav/BASIC5000_{}.wav'.format(str(self.num).zfill(4))
            self.load_wav(file_path,1,i)
            file_path = 'wav2/BASIC5000_{}.wav'.format(str(self.num).zfill(4))
            self.load_wav(file_path,2,i)
            file_path = 'wav3/BASIC5000_{}.wav'.format(str(self.num + 600).zfill(4))
            self.load_wav(file_path,3,i)
            file_path = 'wav456/fujitou_normal/fujitou_normal_{}.wav'.format(str(self.num).zfill(3))
            self.load_wav(file_path,4,i)
            file_path = 'wav456/tsuchiya_normal/tsuchiya_normal_{}.wav'.format(str(self.num).zfill(3))
            self.load_wav(file_path,5,i)
            file_path = 'wav456/uemura_normal/uemura_normal_{}.wav'.format(str(self.num).zfill(3))
            self.load_wav(file_path,6,i)
            self.num = self.num + 4

        self.fitdata()


    def load_wav(self,file_name,data_num,num):
        if (data_num,num) == (1,self.FILESUM-1):
            self.split_len = len(self.ls)-1
            print("split_len:{}".format(self.split_len))
        ls = []
        log_mel_specs = []

        try:
            x, fs = sf.read('./'+ file_name)
        except Exception as e:
            print(e)
            return None
        mels = librosa.feature.melspectrogram(y=x, sr=fs,n_mels = self.n_mels).T
        mel_len , n_mels = (mels.shape)
        #mel = np.ravel(mels)
        mel = mels.tolist()

        #動的特徴量1
        log_mel_specs = mel
        delta_logmel_lists = []
        for i,log_mel_spec in enumerate(log_mel_specs):
            delta_logmel_list = []
            for count_n_logmel,m in enumerate(log_mel_spec):
                if i != 0:
                    delta_logmel = m - log_mel_specs[i-1][count_n_logmel]
                else:
                    delta_logmel = m-m
                delta_logmel_list.append(delta_logmel)
            delta_logmel_lists.append(delta_logmel_list)
        self.delta_feature_lists = self.delta_feature_lists + delta_logmel_lists

        #動的特徴量2
        delta_delta_logmel_lists = []
        for i,delta_logmel_list in enumerate(delta_logmel_lists):
            delta_delta_logmel_list = []
            for count_n_logmel,delta_mfcc in enumerate(delta_logmel_list):
                if i != 0:
                    delta_delta_logmel = delta_logmel_lists[i-1][count_n_logmel]
                else:
                    delta_delta_logmel = m-m
                delta_delta_logmel_list.append(delta_logmel)
            delta_delta_logmel_lists.append(delta_logmel_list)
        self.delta_delta_feature_lists = self.delta_delta_feature_lists + delta_delta_logmel_lists

        self.logmel_lists = self.logmel_lists + mel
        #print(len(self.logmel_lists))
        n,data_sum = (self.n_mels,mel_len)
        #print(data_sum, n)
        self.ls = np.append(self.ls,np.full(data_sum,data_num-1))
        #self.ls = np.append(self.ls,np.full())
        self.length_ls.append(data_sum)


    def fitdata(self):
        #print(self.ls[::100])
        #print(self.length_ls)
        #self.ls = self.ls[1::]


        """
        #標準化
        self.logmel_lists = sklearn.preprocessing.scale(self.logmel_lists, axis=1)
        self.delta_feature_lists = sklearn.preprocessing.scale(self.delta_feature_lists,axis=1)
        self.delta_delta_feature_lists = sklearn.preprocessing.scale(
                self.delta_delta_feature_lists,axis=1)
        print(self.logmel_lists.shape)
        print(self.delta_feature_lists.shape)
        print(self.delta_delta_feature_lists.shape)
        """


        #動的特徴量を追加
        feature = np.append(self.logmel_lists,self.delta_feature_lists,axis=1)
        feature = np.append(feature,self.delta_delta_feature_lists,axis=1)
        #feature = self.logmel_lists
        print("feature:{}".format(feature.shape))
        #print(len(feature))

        #ニューラルネットワークで扱える形に変換
        feature3 = []
        label_ls = []
        tf = True
        number_of_file = len(self.length_ls)
        for count in range(number_of_file-1):
            if (count >= number_of_file - 7)and(tf):
                sp_len = len(label_ls) + 1
                tf = False
            for i in range(0,self.length_ls[count+1]-self.SPLICE_SIZE,self.SPLICE_LANGE):
                num = sum(self.length_ls[:count+1]) + i
                a = feature[num:num+self.SPLICE_SIZE]
                np.ravel(a)
                feature3.append(a)
                l = [self.ls[num]]
                label_ls.extend(l)
                #print(num,i,count,str(tf))

        feature3 = np.array(feature3)
        train_labels = to_categorical(label_ls, self.DATA_SIZE)
        print("feature3:{}".format(feature3.shape))
        print("train_labels:{}".format(train_labels.shape))

        training_accuracy = []
        test_accuracy = []
        print("fit")

        #学習用と検証用に分ける
        #print(sp_len)
        self.x_train = feature3[:sp_len]
        self.x_test = feature3[sp_len:]
        self.label_train = train_labels[:sp_len]
        self.label_test = train_labels[sp_len:]
        print("x_train:{}".format(self.x_train.shape))
        print("x_test:{}".format(self.x_test.shape))
        print("label_train:{}".format(self.label_train.shape))
        print("label_test:{}".format(self.label_test.shape))
        print(self.label_test[::10])

        #NPデータを保存
        a = (feature3,train_labels)
        with open('logmel_cnn_np_{}.pkl'.format(str(self.N)),'wb') as f:
            pickle.dump(a,f)

        #CNNモデル
        model = Sequential()
        model.add(Conv1D(self.SPLICE_SIZE,3, padding='same', input_shape=(self.SPLICE_SIZE,self.n_mels*3), activation='relu'))
        model.add(Conv1D(self.SPLICE_SIZE,3, padding='same', activation='relu'))
        model.add(MaxPooling1D(2, padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv1D(self.SPLICE_SIZE/2,3, padding='same', activation='relu'))
        #model.add(Conv1D(self.SPLICE_SIZE/2,3, padding='same', activation='relu'))
        model.add(MaxPooling1D(2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1D(int(self.SPLICE_SIZE / 4), 3, padding='same', activation='tanh'))

        model.add(Flatten())
        model.add(Dense(196, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.DATA_SIZE, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['acc'])
        print(model.summary())
        history = model.fit(self.x_train, self.label_train, batch_size=64,epochs=32, validation_split=0.2)#32

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

        x = model.predict(self.x_test)
        print(x)
        print(self.label_test)

        #CNNモデルの保存
        with open('logmel_cnn_m.pkl','wb') as f:
            pickle.dump(model,f)
        print('finish')

    def cnn_predict(self,file_name):
        with open('logmel_cnn_m.pkl','rb') as f:
            model = pickle.load(f)
        try:
            x, fs = sf.read('./'+ file_name)
        except Exception as e:
            print(e)
            return None
        mels = librosa.feature.melspectrogram(y=x, sr=fs,n_mels = self.n_mels).T
        mel_len , n_mels = (mels.shape)
        mel = mels.tolist()

        #動的特徴量1
        log_mel_specs = mel
        delta_logmel_lists = []
        for i,log_mel_spec in enumerate(log_mel_specs):
            delta_logmel_list = []
            for count_n_logmel,m in enumerate(log_mel_spec):
                if i != 0:
                    delta_logmel = m - log_mel_specs[i-1][count_n_logmel]
                else:
                    delta_logmel = m-m
                delta_logmel_list.append(delta_logmel)
            delta_logmel_lists.append(delta_logmel_list)

        #動的特徴量2
        delta_delta_logmel_lists = []
        for i,delta_logmel_list in enumerate(delta_logmel_lists):
            delta_delta_logmel_list = []
            for count_n_logmel,delta_mfcc in enumerate(delta_logmel_list):
                if i != 0:
                    delta_delta_logmel = delta_logmel_lists[i-1][count_n_logmel]
                else:
                    delta_delta_logmel = m-m
                delta_delta_logmel_list.append(delta_logmel)
            delta_delta_logmel_lists.append(delta_logmel_list)

        #動的特徴量追加
        feature = np.append(mel,delta_logmel_lists,axis=1)
        feature = np.append(feature,delta_delta_logmel_lists,axis=1)
        print(feature.shape)

        #ニューラルネットワークで扱える形に変換
        feature3 = []
        label_ls = []
        number_of_file = len(self.length_ls)
        for i in range(0,mel_len-self.SPLICE_SIZE):#,self.SPLICE_LANGE):
            num = i
            a = feature[num:num+self.SPLICE_SIZE]
            np.ravel(a)
            feature3.append(a)
            #print(num,i,len(a))

        feature3 = np.array(feature3)
        #print(feature3.shape)
        x = model.predict(feature3)
        ls = [ np.argmax(i) for i in x]
        c = collections.Counter(ls)
        return c.most_common()[0]

    def delta_parameters(self,ls):
        delta_logmel_lists = []
        for i,log_mel_spec in enumerate(ls):
            delta_logmel_list = []
            for count_n_logmel,m in enumerate(log_mel_spec):
                if i != 0:
                    delta_logmel = m - log_mel_specs[i-1][count_n_logmel]
                else:
                    delta_logmel = m-m
                delta_logmel_list.append(delta_logmel)
            delta_logmel_lists.append(delta_logmel_list)
        return delta_logmel_lists



if __name__ == '__main__':
    x = MFCC()
    x.Machine_Learning()
    #print(x.cnn_predict("BASIC5000_0085.wav"))

