import librosa
import numpy as np
import os


def get_feature(file_path: str, mfcc_len: int = 39, mean_signal_length: int = 96000):
    '''
    file_path: Speech signal folder
    mfcc_len: MFCC coefficient length
    mean_signal_length: MFCC feature average length
    :return: MFCC feature
    '''

    signal, fs = librosa.load(file_path)  # 66150 22050
    s_len = len(signal)

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=40)  # 40  128  signal:96000 fs:22050
    mfcc = mfcc.T
    feature = mfcc
    # print(feature.shape)
    # exit()
    return feature


a = []
c = []
# 处理HC文件夹里的MFCC
dir_path = r'D:\speech_data_dir\ReadText_3.0_segment\HC\\'
for dir, subdirs, files in os.walk(dir_path):
    for f in files:
        if f.split('.')[-1] == 'wav':
            mfcc = get_feature(file_path=dir_path + f)

            # print(mfcc.shape)
            # exit()

            a.append(mfcc)
num = len(a)
print(num)

# 处理PD文件夹里的MFCC
dir_path = r'D:\speech_data_dir\ReadText_3.0_segment\PD\\'
for dir, subdirs, files in os.walk(dir_path):
    for f in files:
        if f.split('.')[-1] == 'wav':
            mfcc = get_feature(file_path=dir_path + f)
            a.append(mfcc)
Num = len(a) - num
print(Num)
print(len(a))
print('-'*30)
features = np.array(a)

# 处理One-Hot标签 HC:[1,0] PD:[0,1]
indexs = np.array([0] * num)
for index in indexs:
    label = np.zeros(2, dtype=np.float32)  # 创建具有10个标签的onehot
    label[index] = 1
    c.append(label)

indexs = np.array([1] * Num)
for index in indexs:
    label = np.zeros(2, dtype=np.float32)  # 创建具有10个标签的onehot
    label[index] = 1
    c.append(label)
labels = np.array(c)

target_final = {'x': features, 'y': labels}
print(type(target_final['x']))
print(target_final['x'].shape)

print(type(target_final['y']))
print(target_final['y'].shape)

np.save('../TF-AMSP-main/Code/MFCC/ReadText11.npy', target_final)
