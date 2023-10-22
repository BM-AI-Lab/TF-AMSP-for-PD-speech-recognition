"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
import numpy as np
import keras.backend as K
import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from keras import callbacks
from keras.layers import Layer, Dense, Input
from keras.models import Model

from Common_Model import Common_Model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime  # datetime.datetime.now()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import heatmap

from TIMNET import TIMNET


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels = labels * (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='uniform',
                                      trainable=True)
        super(WeightLayer, self).build(input_shape)

    def call(self, x):
        tempx = tf.transpose(x, [0, 2, 1])  # (None,39,188)
        x = K.dot(tempx, self.kernel)
        x = tf.squeeze(x, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


class TIMNET_Model(Common_Model):
    def __init__(self, args, input_shape, class_label, **params):
        super(TIMNET_Model, self).__init__(**params)
        self.args = args
        self.data_shape = input_shape  # （188， 39）
        self.num_classes = len(class_label)  # 2
        self.class_label = class_label  # （'hc', 'pd'）
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        print("TIMNET MODEL SHAPE:", input_shape)
    
    def create_model(self):
        self.inputs=Input(shape=(self.data_shape[0], self.data_shape[1]))

        self.multi_decision = TIMNET(nb_filters=self.args.filter_size,
                                kernel_size=self.args.kernel_size, 
                                nb_stacks=self.args.stack_size,
                                dilations=self.args.dilation_size,
                                dropout_rate=self.args.dropout,
                                activation =self.args.activation,
                                return_sequences=True, 
                                name='TIMNET')(self.inputs)  # (None, 8, 39)

        self.model = Model(inputs=self.inputs, outputs=self.multi_decision)
        
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2, epsilon=1e-8),
                           metrics=['accuracy'])
        print("Temporal create succes!")
        
    def train(self, x, y):

        filepath = self.args.model_path  # './Models'
        resultpath = self.args.result_path  # './Results'

        if not os.path.exists(filepath):
            os.mkdir(filepath)
        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        i = 1
        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)  # 留一交叉验证
        avg_accuracy = 0
        avg_loss = 0
        for train, test in kfold.split(x, y):  # Generate indices to split data into training and test set.
            self.create_model()
            y[train] = smooth_labels(y[train], 0.1)
            folder_address = filepath+self.args.data+"_"+str(self.args.random_seed)+"_"+now_time
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
            weight_path = folder_address+'/'+str(self.args.split_fold)+"-fold_weights_best_"+str(i)+".hdf5"
            checkpoint = callbacks.ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=1, save_weights_only=True,
                                                   save_best_only=True, mode='max')  # save_best_only
            self.model.fit(x[train], y[train], validation_data=(x[test],  y[test]), batch_size=self.args.batch_size,
                               epochs=self.args.epoch, verbose=1, callbacks=[checkpoint])

            self.model.load_weights(weight_path)
            best_eva_list = self.model.evaluate(x[test],  y[test])  # best_eva_list = (avg_loss, avg_accuracy)
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]

            print(str(i)+'_Model evaluation: ', best_eva_list, "   Now ACC:", str(round(avg_accuracy*10000)/100/i))
            i += 1
            y_pred_best = self.model.predict(x[test])
            self.matrix.append(confusion_matrix(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1)))
            em = classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label, output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label, digits=4))

        print("Average ACC:", avg_accuracy/self.args.split_fold)
        self.acc = avg_accuracy/self.args.split_fold
        writer = pd.ExcelWriter(resultpath+self.args.data+'_'+str(self.args.split_fold)+'fold_'+str(round(self.acc*10000) / 100)+"_"+str(self.args.random_seed)+"_"+now_time+'.xlsx')
        for i, item in enumerate(self.matrix):
            temp = {}
            temp[" "] = self.class_label
            for j, l in enumerate(item):
                temp[self.class_label[j]] = item[j]
            data1 = pd.DataFrame(temp)
            data1.to_excel(writer, sheet_name=str(i), encoding='utf8')

            df = pd.DataFrame(self.eva_matrix[i]).transpose()
            df.to_excel(writer, sheet_name=str(i)+"_evaluate", encoding='utf8')
        writer.save()
        writer.close()

        K.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.trained = True
    
    def test(self, x, y, path):
        i = 1
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        x_feats = []
        y_labels = []
        confusion_matrix_all = 0
        for train, test in kfold.split(x, y):
            self.create_model()
            weight_path=path+'/'+str(self.args.split_fold)+"-fold_weights_best_"+str(i)+".hdf5"
            self.model.fit(x[train], y[train], validation_data=(x[test],  y[test]), batch_size=64, epochs=0, verbose=0)
            self.model.load_weights(weight_path)  # +source_name+'_single_best.hdf5')
            best_eva_list = self.model.evaluate(x[test],  y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i)+'_Model evaluation: ', best_eva_list, "   Now ACC:", str(round(avg_accuracy*10000)/100/i))
            i += 1
            y_pred_best = self.model.predict(x[test])
            self.matrix.append(confusion_matrix(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1)))

            # TODO
            # 绘制混淆矩阵
            a = confusion_matrix(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1))
            confusion_matrix_all += a

            em = classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label,output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label, digits=4))
            caps_layer_model = Model(inputs=self.model.input,
            outputs=self.model.get_layer(index=-2).output)
            feature_source = caps_layer_model.predict(x[test])
            x_feats.append(feature_source)
            y_labels.append(y[test])

        # TODO
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        cmap = sns.light_palette("green", as_cmap=True)
        sns.heatmap(confusion_matrix_all, annot=True, fmt='d', cmap=cmap,
                    xticklabels=['HC', 'PD'],
                    yticklabels=['HC', 'PD'],
                    annot_kws={"size": 28, "color": "black"},
                    cbar=False)  # "fontweight": "bold",

        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)

        plt.xlabel('Predicted', fontsize=28)  # 调整 x 轴标签的字体大小
        plt.ylabel('True', fontsize=28)  # 调整 y 轴标签的字体大小
        plt.title('Confusion Matrix', fontsize=36)  # 调整标题的字体大小

        # 保存混淆矩阵图
        plt.savefig('./confusion_matrix/confusion_matrix.png')

        # 显示图像
        plt.show()

        print("Average ACC:", avg_accuracy/self.args.split_fold)
        self.acc = avg_accuracy/self.args.split_fold

        # TODO 算一下最后的平均结果
        all_mp = 0
        all_wp = 0
        all_mr = 0
        all_wr = 0
        all_mf1 = 0
        all_wf1 = 0
        for eval_matrix in self.eva_matrix:
            mp = eval_matrix['macro avg']['precision']
            wp = eval_matrix['weighted avg']['precision']
            mr = eval_matrix['macro avg']['recall']
            wr = eval_matrix['weighted avg']['recall']
            mf1 = eval_matrix['macro avg']['f1-score']
            wf1 = eval_matrix['weighted avg']['f1-score']

            all_mp += mp
            all_wp += wp
            all_mr += mr
            all_wr += wr
            all_mf1 += mf1
            all_wf1 += wf1

        all_mp /= 10
        all_wp /= 10
        all_mr /= 10
        all_wr /= 10
        all_mf1 /= 10
        all_wf1 /= 10

        print("macro avg precision: {:.4f}, macro avg recall: {:.4f}, macro avg f1_score: {:.4f}".format(all_mp, all_wr, all_mf1))
        print("weighted avg precision: {:.4f}, weighted avg recall: {:.4f}, weighted avg f1_score: {:.4f}".format(all_wp, all_wr, all_wf1))

        return x_feats, y_labels
