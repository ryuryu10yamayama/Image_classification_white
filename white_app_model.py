import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras import optimizers
from keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler

img_Baymax = []
img_Kodama = []
img_Olaf = []
img_Snow = []


#255で割ることで正規化しています

#Baymax
Baymax_files = glob.glob("./Baymax/*.jpg")
for file in Baymax_files:
    img = cv2.imread(file)
    img = img_to_array(img) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img_Baymax.append(img)

#Kodama
Kodama_files = glob.glob("./Kodama/*.jpg")
for file in Kodama_files:
    img = cv2.imread(file)
    img = img_to_array(img) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img_Kodama.append(img)

#Olaf
Olaf_files = glob.glob("./Olaf/*.jpg")
for file in Olaf_files:
    img = cv2.imread(file)
    img = img_to_array(img) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img_Olaf.append(img)

#Snow Man
Snow_files = glob.glob("./Snow Man/*.jpg")
for file in Snow_files:
    img = cv2.imread(file)
    img = img_to_array(img) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img_Snow.append(img)

#画像データの結合とラベルの作成
X = np.array(img_Baymax+img_Kodama+img_Olaf+img_Snow)
y = np.array([0]*len(img_Baymax) + [1]*len(img_Kodama) +
             [2]*len(img_Olaf) + [3]*len(img_Snow))

#画像の順番をランダムにする
rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
y = y[rand_index]

#trainデータとtestデータを作成する
X_train = X[:int(len(X)*0.6)]
X_val = X[int(len(X)*0.6):int(len(X)*0.8)]
y_train = y[:int(len(y)*0.6)]
y_val = y[int(len(y)*0.6):int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]


#labelに関しては4種類あるので、one-hot表現に変換する
y_train = to_categorical(y_train, 4)
y_val = to_categorical(y_val, 4)
y_test = to_categorical(y_test, 4)


#vgg16を用いた転移学習を行う
#元の1000分類に使用するわけではないのでinclude_top=Falseとして特徴抽出器として使用する
input_tensor = Input(shape=(150, 150, 3))
vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor)


#include_top=Falseとした場合のoutputは4次元
#平滑化の際に必要なinputは3次元
#reluは特徴を強調してくれる
#Dropoutで汎化性能の向上
#ネットワークの大きさの調整はここで層を増やしたり、ノードを減らしたりといった操作をする
pr_model = Sequential()
pr_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
pr_model.add(Dense(256, activation="relu"))
pr_model.add(BatchNormalization())
pr_model.add(Dropout(0.7))
pr_model.add(Dense(4, activation="softmax"))

#モデルの連結
model = Model(inputs=vgg16.input, outputs=pr_model(vgg16.output))

#層の固定（どこのMaxPoolingから使用するか、今回の分類対象はcifar10には存在しないので、学習する幅を広げるために10層までを固定する）
for layer in model.layers[:11]:
    layer.trainable = False


#Early-stopping実装(過学習が進んでいくとVal_errorは大きくなっていくので、途中でstopさせる)
#patienceはval_lossが下がらなくなった時点からどのくらい様子を見るか
early_stopping = EarlyStopping(patience=2, verbose=1)


#多クラス分類のため損失関数はcategorical_crossentropyを使用
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(
    lr=0.0001, decay=0.001), metrics=['accuracy'])


#epochs
history = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1,
                    validation_data=(X_val, y_val), callbacks=[early_stopping])
scores = model.evaluate(X_test, y_test, batch_size=32, verbose=0)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#過学習しているかどうか

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('精度を示すグラフのファイル名')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('損失値を示すグラフのファイル名')
