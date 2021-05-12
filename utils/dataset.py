import os
import json
import random
import math
import tensorflow as tf
import numpy as np
from tqdm import tqdm

class SequenceGenerator(tf.keras.utils.Sequence) :
    def __init__(self, songs, batch_size, sequence_length, padding_token, shuffle=True, augmented=True) :
        self.songs = songs
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.padding_token = padding_token
        self.shuffle = shuffle
        self.augmented = augmented
        self.on_epoch_end()
    
    def __len__(self) :
        return int(np.ceil(len(self.songs)/self.batch_size))

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.songs)
            
    def get_closest(self, value, values):
      result = None
      best_diff = math.inf
      for i, v in enumerate(values):
        diff = abs(value - v)
        if diff < best_diff:
          best_diff = diff
          result = i 

      return result
            
    def augment(self, song):
        augmented_song = song
        pitches = [-3, -2, -1, 0, 1, 2, 3]
        time_stretches = [0.95, 0.975, 1.0, 1.025, 1.05]

        offset_pitch = pitches[np.random.randint(0, len(pitches))]
        time_stretch = time_stretches[np.random.randint(0, len(time_stretches))]
        time_shifts = [i for i in range(10, 1001, 10)]
        for i, token in enumerate(augmented_song):
            # Note on
            if 0 <= token < 128:
                augmented_song[i] = min(max(token + offset_pitch, 0), 127)
                
            # Note off
            elif 128 <= token < 256:
                augmented_song[i] = min(max(token + offset_pitch, 128), 255)
                
            # Time shift
            elif 288 <= token < 388:
                ts = time_shifts[token - 288]
                result = self.get_closest(ts * time_stretch, time_shifts)
                augmented_song[i] = result + 288
        
        return augmented_song
        
    def __getitem__(self, index):
        inputs = np.empty((0, self.sequence_length), float)
        targets = np.empty((0, self.sequence_length), float)
        for i in range(self.batch_size):
            if index * self.batch_size + i >= len(self.songs) - 1:
                if len(inputs) > 0:
                    return inputs, targets

            song = self.songs[index * self.batch_size + i]
            
            end = len(song) - self.sequence_length - 1
            start_index = 0 if end < 0 else random.randint(0, len(song) - self.sequence_length - 1)
            
            sequence = song[start_index:start_index + self.sequence_length + 1]
            if self.augmented:
                sequence = self.augment(sequence)
            sequence_input = sequence[:-1]
            sequence_target = sequence[1:]
            
            if len(sequence_target) < self.sequence_length:
                padding = self.sequence_length - len(sequence_target)
                sequence_input = np.pad(sequence_input, (0, padding), 'constant', constant_values=(self.padding_token))
                sequence_target = np.pad(sequence_target, (0, padding), 'constant', constant_values=(self.padding_token))
            
            inputs = np.append(inputs, [sequence_input], axis=0)
            targets = np.append(targets, [sequence_target], axis=0)
            
        return inputs, targets


def load_dataset(dataset):
    data = []
    path = f'{dataset}/data.npy'
    with open(path, 'rb') as f:
        data = np.load(f, allow_pickle=True)
    
    print(f'Using dataset \'{dataset}.\'')
    
    return data


def create_generator_and_validation(data, split):
    with open('parameters.json') as f:
        params = json.load(f)

    padding_token = params['vocab_size'] - 1
    split_index = int(len(data)*split)
    train_split = SequenceGenerator(data[:-split_index], params['batch_size'], params['sequence_length'], padding_token)
    validation_split = SequenceGenerator(data[-split_index:], 1, params['sequence_length'], padding_token, augmented=False)

    x_val = []
    y_val = []
    for x, y in validation_split:
        x_val.append(x[0])
        y_val.append(y[0])
    validation = (np.array(x_val), np.array(y_val))

    return train_split, validation, params['batch_size']
