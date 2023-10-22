import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

x_feats = np.load('x_feats.npy', allow_pickle=True)
y_labels = np.load('y_labels.npy', allow_pickle=True)

for x_feat, y_label in x_feats, y_labels:
    # 将 one-hot 编码转换为类别索引
    y_true = np.argmax(y_labels, axis=1)
    y_pred = np.argmax(x_feats, axis=1)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1', 'Class 2'],
                yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # 保存混淆矩阵图
    plt.savefig('confusion_matrix.png')

    # 显示图像
    plt.show()
