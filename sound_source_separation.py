import numpy as np
#from sklearn.decomposition import NMF
import sounddevice as sd
import wave
import librosa
import soundfile as sf

class separation():

    def __init__(self):
        self.num = 5

    def main(self):

        file_path = 'jsut_ver1_1/basic5000/wav/BASIC5000_{}.wav'.format(str(1).zfill(4))
        S_d = self.load_wav(file_path)
        file_path = 'wave_file/test_add{}.wav'.format(str(self.num))
        S_db = self.load_wav(file_path)

        # ドラム音学習で分解する基底数
        R_d = 50
        # ドラム音学習で行う反復計算回数
        n_iter=300
        #nmf = NMF(n_components = 50)
        nmf_d = self.NMF(np.abs(S_d), R=R_d, n_iter=n_iter)
        #nmf_d = nmf.fit_transform(S_d)#NMF(S_d)#np.abs(S_d), R=R_d, n_iter=n_iter)
        base = nmf_d[0]

        # Semi-Supervised NMF で分解する基底数
        R = 20
        # Semi-Supervised NMF で行う反復計算回数
        n_iter = 50
        ssnmf_db_d = self.SSNMF(np.abs(S_db), F=base, R=R, n_iter=n_iter)

        remixed_only_d = librosa.istft((np.dot(ssnmf_db_d[0], ssnmf_db_d[1]) * (np.cos(np.angle(S_db) + 1j * np.sin(np.angle(S_db))))))

        remixed_without_d = librosa.istft(np.dot(ssnmf_db_d[2], ssnmf_db_d[3]) * (np.cos(np.angle(S_db) + 1j * np.sin(np.angle(S_db)))))

        save_data1 = remixed_only_d
        save_data2 = remixed_without_d
        print(save_data1.shape)

        save_data1 = save_data1 / save_data1.max() * np.iinfo(np.int16).max
        save_data1 = save_data1.astype(np.int16)
        save_data2 = save_data2 / save_data2.max() * np.iinfo(np.int16).max
        save_data2 = save_data2.astype(np.int16)

        with wave.open('./wave_file/test{}.wav'.format(str(self.num)), mode='w') as wb:
            wb.setnchannels(1)  # モノラル
            wb.setsampwidth(2)  # 16bit=2byte
            wb.setframerate(48000)
            wb.writeframes(save_data1.tobytes())  # バイト列に変換
        with wave.open('./wave_file/test{}_1.wav'.format(str(self.num)), mode='w') as wb:
            wb.setnchannels(1)  # モノラル
            wb.setsampwidth(2)  # 16bit=2byte
            wb.setframerate(48000)
            wb.writeframes(save_data2.tobytes())  # バイト列に変換



    def load_wav(self,file_name):
        #音声ファイルのロード
        try:
            x, fs = sf.read('./'+ file_name)#sf.read('./'+ file_name)
            #x, fs = librosa.load('./'+ file_name)#sf.read('./'+ file_name)
        except Exception as e:
            print(e)
            return None
        print(x.shape,fs)
        S_db = librosa.stft(x)
        return S_db
        """
        #logmel
        mels = librosa.feature.melspectrogram(y=x, sr=fs,n_mels = self.n_mels).T
        mel_len , n_mels = (mels.shape)
        mel = mels.tolist()
        return mel
        """

    def mel_nmf(self):
        nmf = NMF(n_components=3)
        W = nmf.fit_transform(X)
        H = nmf.components_

    def SSNMF(self,Y, R=3, n_iter=50, F=[], init_G=[], init_H=[], init_U=[], verbose=False):
        """
        decompose non-negative matrix to components and activation with Semi-Supervised NMF

        Y ≈　FG + HU
        Y ∈ R (m, n)
        F ∈ R (m, x)
        G ∈ R (x, n)
        H ∈ R (m, k)
        U ∈ R (k, n)

        parameters
        ----
        Y: target matrix to decompose
        R: number of bases to decompose
        n_iter: number for executing objective function to optimize
        F: matrix as supervised base components
        init_W: initial value of W matrix. default value is random matrix
        init_H: initial value of W matrix. default value is random matrix

        return
        ----
        Array of:
        0: matrix of F
        1: matrix of G
        2: matrix of H
        3: matrix of U
        4: array of cost transition
        """

        eps = np.spacing(1)

        # size of input spectrogram
        M = Y.shape[0];
        N = Y.shape[1];
        X = F.shape[1]
        # initialization
        if len(init_G):
            G = init_G
            X = init_G.shape[1]
        else:
            G = np.random.rand(X, N)

        if len(init_U):
            U = init_U
            R = init_U.shape[0]
        else:
            U = np.random.rand(R, N)

        if len(init_H):
            H = init_H;
            R = init_H.shape[1]
        else:
            H = np.random.rand(M, R)

        # array to save the value of the euclid divergence
        cost = np.zeros(n_iter)

        # computation of Lambda (estimate of Y)
        Lambda = np.dot(F, G) + np.dot(H, U)

        # iterative computation
        for it in range(n_iter):

            # compute euclid divergence
            cost[it] = self.euclid_divergence(Y, Lambda + eps)

            # update of H
            H *= (np.dot(Y, U.T) + eps) / (np.dot(np.dot(H, U) + np.dot(F, G), U.T) + eps)

            # update of U
            U *= (np.dot(H.T, Y) + eps) / (np.dot(H.T, np.dot(H, U) + np.dot(F, G)) + eps)

            # update of G
            G *= (np.dot(F.T, Y) + eps)[np.arange(G.shape[0])] / (np.dot(F.T, np.dot(H, U) + np.dot(F, G)) + eps)

            # recomputation of Lambda (estimate of V)
            Lambda = np.dot(H, U) + np.dot(F, G)

        return [F, G, H, U, cost]

    def euclid_divergence(self, V, Vh):
        d = 1 / 2 * (V ** 2 + Vh ** 2 - 2 * V * Vh).sum()
        return d

    def NMF(self, Y, R=3, n_iter=50, init_H=[], init_U=[], verbose=False):
        """
        decompose non-negative matrix to components and activation with NMF
        
        Y ≈　HU
        Y ∈ R (m, n)
        H ∈ R (m, k)
        HU ∈ R (k, n)
        
        parameters
        ---- 
        Y: target matrix to decompose
        R: number of bases to decompose
        n_iter: number for executing objective function to optimize
        init_H: initial value of H matrix. default value is random matrix
        init_U: initial value of U matrix. default value is random matrix
        
        return
        ----
        Array of:
        0: matrix of H
        1: matrix of U
        2: array of cost transition
        """

        eps = np.spacing(1)

        # size of input spectrogram
        M = Y.shape[0]
        N = Y.shape[1]
        
        # initialization
        if len(init_U):
            U = init_U
            R = init_U.shape[0]
        else:
            U = np.random.rand(R,N);

        if len(init_H):
            H = init_H;
            R = init_H.shape[1]
        else:
            H = np.random.rand(M,R)

        # array to save the value of the euclid divergence
        cost = np.zeros(n_iter)

        # computation of Lambda (estimate of Y)
        Lambda = np.dot(H, U)

        # iterative computation
        for i in range(n_iter):

            # compute euclid divergence
            cost[i] = self.euclid_divergence(Y, Lambda)

            # update H
            H *= np.dot(Y, U.T) / (np.dot(np.dot(H, U), U.T) + eps)
            
            # update U
            U *= np.dot(H.T, Y) / (np.dot(np.dot(H.T, H), U) + eps)
            
            # recomputation of Lambda
            Lambda = np.dot(H, U)

        return [H, U, cost]

    def euclid_divergence(self, Y, Yh):
        d = 1 / 2 * (Y ** 2 + Yh ** 2 - 2 * Y * Yh).sum()
        return d

if __name__ == '__main__':
    separation().main()
