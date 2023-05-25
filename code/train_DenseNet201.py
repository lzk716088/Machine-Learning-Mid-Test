import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input

# 設定圖片大小
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 100

################### 資料前處理 ###################

# 載入訓練資料
train_df = pd.read_csv(r"train.csv")
train_df['Label'] = train_df['Label'].astype(str)
print(train_df.head())

# 設定訓練影像生成器
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=r"C:\Users\aiialab\Desktop\siang\aoi\train_images\train_images",
    x_col='ID',
    y_col='Label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=False
)
validation_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=r"C:\Users\aiialab\Desktop\siang\aoi\train_images\train_images",
    x_col='ID',
    y_col='Label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 載入測試資料
test_df = pd.read_csv(r"test.csv")
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=r"C:\Users\aiialab\Desktop\siang\aoi\test_images\test_images",
    x_col='ID',
    y_col=None,
    target_size=IMAGE_SIZE,
    batch_size=22,
    class_mode=None,
    shuffle=False
)

################### 模型訓練 ###################
def DenseNet_create_and_fit_model(train_generator, validation_generator, summary=True, fit=True, epochs=EPOCHS):
    model = Sequential()
    conv_base = DenseNet201(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    conv_base.trainable = False

    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", 
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "acc"])
    
    callbacks = [EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="min"), 
             ModelCheckpoint(filepath="mymodel_DenseNet.h5", monitor="val_loss", mode="min", save_best_only=True, save_weights_only=False, verbose=1)]


    if summary:
        model.summary()

    if fit:
        history = model.fit_generator(generator=train_generator, epochs=epochs, validation_data=validation_generator, 
                                    callbacks=callbacks, workers=4, steps_per_epoch=10, validation_steps=251//BATCH_SIZE)
    return model, history

# 建立並訓練模型
model, history = DenseNet_create_and_fit_model(train_generator, validation_generator)

# 載入模型
model = tf.keras.models.load_model('mymodel_DenseNet.h5')

def CNN_model_evaluate(model):
    loss, precision, recall, acc = model.evaluate(validation_generator, batch_size=22)
    print("Test Accuracy: %.2f" % (acc))
    print("Test Loss: %.2f" % (loss))
    print("Test Precision: %.2f" % (precision))
    print("Test Recall: %.2f" % (recall))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history["acc"], color="r", label="Training Accuracy")
    plt.plot(history.history["val_acc"], color="b", label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.ylim([min(plt.ylim()),1])
    plt.title("Training and Validation Accuracy", fontsize=16)
    plt.savefig("DenseNet_acc.png")
    plt.close()
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], color="r", label="Training Loss")
    plt.plot(history.history["val_loss"], color="b", label="Validation Loss")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.ylim([0, max(plt.ylim())])
    plt.title("Training and Validation Loss", fontsize=16)
    plt.savefig("DenseNet_Loss.png")
    plt.close()
CNN_model_evaluate(model)

################### 預測並寫入CSV檔案 ###################

# 預測測試集
test_pred = model.predict(test_generator, verbose=1)

# 取預測結果中最大值的索引作為預測的類別
test_pred = np.argmax(test_pred, axis=1)

# 將預測結果寫入CSV檔案
results_df = pd.DataFrame({'ID': test_df.ID, 'Label': test_pred})
results_df.to_csv('test.csv', index=False)

