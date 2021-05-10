import os
import mido
import math
import sys
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict 

note_ons = [i for i in range(128)]
note_offs = [i for i in range(128)]
velocities = [i for i in range(0, 128, 4)]
time_shifts = [i for i in range(10, 1001, 10)]
programs = [i for i in range(128)]

def get_closest(value, values):
    result = -1
    best_diff = math.inf
    for i, v in enumerate(values):
        diff = abs(value - v)
        if diff < best_diff:
            best_diff = diff
            result = i 

    return result


def create_token(index, velocity=False, timeshift=False, note_on=False, note_off=False, program=False):
    on_offset = 0
    off_offset = len(note_offs)
    vel_offset = off_offset + len(note_ons)
    ts_offset = vel_offset + len(velocities)
    prog_offset = ts_offset + len(time_shifts)

    if note_on:
        return on_offset + index
    elif note_off:
        return off_offset + index
    elif velocity:
        return vel_offset + index
    elif timeshift:
        return ts_offset + index
    elif program:
        return prog_offset + index


def convert_to_tokens(midi_path):
    tokenized = []

    data = mido.MidiFile(midi_path)

    current_velocity = 0
    current_channel = 0
    current_programs = defaultdict(lambda : 0)
    tempo = 500000

    for msg in data:
        if msg.is_meta:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                continue

        time = mido.second2tick(msg.time, data.ticks_per_beat, tempo)

        while time >= 5:
            ts_index = get_closest(time, time_shifts)
            ts = time_shifts[ts_index]
            time -= ts
            tokenized.append(create_token(ts_index, timeshift=True))

        if msg.type == 'program_change':
            current_programs[msg.channel] = msg.program

        if msg.type in { 'note_on', 'note_off' }:
            vel = msg.velocity
            note = msg.note

            if msg.channel != current_channel:
                current_channel = msg.channel
                tokenized.append(create_token(current_programs[current_channel], program=True))

            if vel == 0 or msg.type == 'note_off':
                tokenized.append(create_token(note, note_off=True))
            else:
                if current_velocity != vel:
                    vel_index = get_closest(vel, velocities)
                    current_velocity = velocities[vel_index]
                    tokenized.append(create_token(vel_index, velocity=True))

                tokenized.append(create_token(note, note_on=True))

    return tokenized


def process_midis(midi_dir, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    processed = []
    skipped = 0
    min_size = 10000
    for midi in tqdm(os.listdir(midi_dir), 'converting'):
        path = f'{midi_dir}/{midi}'

        file_size = os.path.getsize(path)
        if file_size < min_size:
            skipped += 1
            continue

        try:
            res = convert_to_tokens(path)
        except:
            skipped += 1
            continue

        processed.append(res)
        
    with open(f'{output_path}/data.npy', 'wb') as f:
        np.save(f, np.array(processed, dtype=object))
        print(f'Skipped {skipped} files.')
        print(f'Saved to \'{output_path}\'.')


def main():
    parser = argparse.ArgumentParser(description='Preprocess a MIDI dataset.')
    parser.add_argument('-i', '--input', type=str, help='the directory containing the MIDIs to be preprocessed')
    parser.add_argument('-o', '--output', type=str, help='the directory for the output')

    args = parser.parse_args()
    process_midis(args.input, args.output)

if __name__ == "__main__":
    main()
