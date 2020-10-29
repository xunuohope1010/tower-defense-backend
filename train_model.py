import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


wave_files = []
folder_list = os.listdir('./')
wave_folder_list = []
height = 8
width = 12
input_shape = (height*width*4+1,)
os.mkdir('weight')

for folder in folder_list:
    if 'wave' in folder:
        wave_folder_list.append(folder)

for wave_folder in wave_folder_list:
    for name in os.listdir('./' + wave_folder + '/'):
        with open('./' + wave_folder + '/' + name, "r") as file:
            wave_files.append(file.read())
game_map_wave_list = []
reward_list = []
wave_number_list = []
for text in wave_files:
    my_json = json.loads(text)
    game_map_1d = np.array(my_json['game_map']).reshape((height, width, 4)).reshape(-1)
    game_map_wave_list.append(np.append(game_map_1d, my_json['wave_number'])/20)
    reward_list.append([my_json['reward']])
    wave_number_list.append(my_json['wave_number'])

divide = 0.9
game_map_length = int(len(game_map_wave_list) * divide)
train_data = np.array(game_map_wave_list[:game_map_length])
train_label = np.array(reward_list[:game_map_length])
test_data = np.array(game_map_wave_list[game_map_length + 1:])
test_label = np.array(reward_list[game_map_length + 1:])

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', use_bias=True, input_shape=input_shape))
model.add(layers.Dense(128, activation='relu', use_bias=True))
model.add(layers.Dense(128, activation='relu', use_bias=True))
model.add(layers.Dense(128, activation='relu', use_bias=True))
model.add(layers.Dense(128, activation='relu', use_bias=True))
model.add(layers.Dense(128, activation='relu', use_bias=True))
model.add(layers.Dense(128, activation='relu', use_bias=True))
model.add(layers.Dense(128, activation='relu', use_bias=True))
model.add(layers.Dense(128, activation='relu', use_bias=True))
model.add(layers.Dense(128, activation='relu', use_bias=True))
model.add(layers.Dense(1, use_bias=True))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=[coeff_determination])

model.summary()
checkpoint_name = 'weight/Weights_{epoch:03d}_{val_coeff_determination:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

history = model.fit(train_data, train_label, epochs=30,
                    validation_data=(test_data, test_label), callbacks=callbacks_list)

# model.save_weights('weight.h5')



