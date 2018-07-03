from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# 读取训练集和测试集
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# 特征
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# 构建DNN网络，3层，每层分别为10,20,10个节点
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[3],
                                            n_classes=3,
                                            optimizer=tf.train.AdamOptimizer(0.05))

# 拟合模型，迭代20000步
classifier.fit(x=training_set.data, y=training_set.target, steps=1000)

# 计算精度
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))

# 预测新样本的类别
new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))
