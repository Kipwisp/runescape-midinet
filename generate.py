import tensorflow as tf 
import numpy as np
import random
import mido
import math
import json
import argparse
import os
import utils.dataset as dataset_utils
from tqdm import tqdm
from datetime import datetime

note_ons = [i for i in range(128)]
note_offs = [i for i in range(128)]
velocities = [i for i in range(0, 128, 4)]
time_shifts = [i for i in range(10, 1001, 10)]
programs = [i for i in range(128)]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def generate_notes(model, input_, song_length, k, inclusive, padding_token):
    seq_length = len(input_)
    result = input_
    for i in tqdm(range(song_length), desc='generating'):
        x = np.array([result[i:i+seq_length]])
        predictions = model.predict(x)
        predictions = predictions[0][-1]
        predictions, indices = tf.math.top_k(predictions, k=k, sorted=True)
        
        predictions = np.asarray(predictions).astype("float32")
        indices = np.asarray(indices).astype("int32")

        if padding_token in indices:
            pos = np.where(indices == padding_token)[0].item()
            indices = np.delete(indices, pos)
            predictions = np.delete(predictions, pos)

        note = np.random.choice(indices, p=softmax(predictions))
        result.append(note)
    
    if inclusive:
        return result
    else:
        return result[seq_length:]


def create_midi(input_, path='output'):
    NOTE_ON, NOTE_OFF, VELOCITY, TIME_SHIFT, PROGRAM_CHANGE, PADDING = 0, 1, 2, 3, 4, 5

    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)

    if not os.path.exists(path):
        os.mkdir(path)

    current_vel = 0
    current_time = 0
    lengths = (len(note_ons), len(note_offs), len(velocities), len(time_shifts), len(programs), 1)
    for action in tqdm(input_, 'creating file'):
        index = action
        for type, length in enumerate(lengths):
            if index >= length:
                index -= length
            else:
                break

        if type == NOTE_ON:
            message = mido.Message('note_on', note=note_ons[index], velocity=current_vel, time=current_time)
            track.append(message)
            current_time = 0
        elif type == NOTE_OFF:
            message = mido.Message('note_off', note=note_offs[index], time=current_time)
            track.append(message)
            current_time = 0
        elif type == VELOCITY:
            current_vel = velocities[index]
        elif type == TIME_SHIFT:
            current_time += time_shifts[index]
        elif type == PROGRAM_CHANGE:
            message = mido.Message('program_change', channel=0, program=index, time=0)
            track.append(message)

    file_name = f'{path}/{datetime.now().strftime("%d-%b-%Y_%H.%M.%S")}.mid'
    midi.save(file_name)
    print(f'Generated MIDI file saved as \'{file_name}\'.')


def generate(song_length, top_k, save_dir, dataset, inclusive):
    with open('parameters.json') as f:
        params = json.load(f)

    try:
        model = tf.keras.models.load_model(save_dir)
    except:
        print(f'Could not find the saved model \'{save_dir}\'.')
        exit()

    data = dataset_utils.load_dataset(dataset)
    padding_token = params['vocab_size'] - 1

    starting_length = 512
    random_song = random.choice(data)
    starting = random_song[:starting_length]
    starting = [padding_token for i in range(params['sequence_length'] - starting_length)] + starting

    result = generate_notes(model, starting, song_length, top_k, inclusive, padding_token)
    create_midi(result)


def main():
    parser = argparse.ArgumentParser(description='Generate a MIDI file using a model.')
    parser.add_argument('-l', '--length', default=2500, type=int, help='the number of tokens to be generated')
    parser.add_argument('-k', '--top_k', default=8, type=int, help='k value for selecting top K predictions')
    parser.add_argument('-s', '--save_directory', default='midinet_model', type=str, help='the directory to load the model from')
    parser.add_argument('-d', '--dataset', type=str, help='the directory containing the dataset to draw a starting seed from')
    parser.add_argument('--inclusive', action='store_true', help='include the seed data in the MIDI file')

    args = parser.parse_args()
    generate(args.length, args.top_k, args.save_directory, args.dataset, args.inclusive)


if __name__ == "__main__":
    main()
