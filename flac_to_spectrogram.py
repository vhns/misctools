#!/usr/bin/env python3

import argparse
import csv
import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_flac(file_path, sample_rate):
    flac, _ = librosa.load(file_path, sr=sample_rate)
    return flac

def compute_mel_spectrogram(file, sample_rate, n_mels, hop_length):
    mel_spectrogram = librosa.feature.melspectrogram(y=file, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    return mel_spectrogram

def save_mel_spectrogram(mel_spectrogram, output_path, sample_rate, hop_length, x_axis=None, y_axis=None, axis='off'):
    np_path = f'{output_path}.npy'
    png_path = f'{output_path}.png'    

    # This has been written as a janky fix for a situation during which the script was stopped halfway through.
    # It checks for the presence of both the numpy and png file.
    if os.path.isfile(np_path) and os.path.isfile(png_path):
        pass
    else:
        np.save(np_path, mel_spectrogram)

        plt.figure(figsize=(5,5))
        image = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(image, sr=sample_rate, hop_length=hop_length, x_axis=None, y_axis=None)
        if axis == 'off':
            plt.axis('off')

        plt.savefig(png_path, bbox_inches='tight', pad_inches=0, format='png')
        plt.close()

def process_flac_file(file_path, sample_rate, n_mels, hop_length, output_path, x_axis, y_axis, axis):
    flac = load_flac(file_path, sample_rate)
    mel_spectrogram = compute_mel_spectrogram(flac, sample_rate, n_mels, hop_length)
    save_mel_spectrogram(mel_spectrogram, output_path, sample_rate, hop_length, x_axis, y_axis, axis)

def process_flac_files(csv_file, delimiter, position, output_dir, sample_rate, n_mels, hop_length, x_axis, y_axis, axis):
    with open(csv_file, mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter)
        for words in reader:
            file_path = words[position]
            filename = Path(file_path).stem
            output_path = os.path.join(output_dir, filename)
            np_path = f'{output_path}.npy'
            png_path = f'{output_path}.png'    

            # This has been written as a janky fix for a situation during which the script was stopped halfway through.
            # It checks for the presence of both the numpy and png file.
            if os.path.isfile(np_path) and os.path.isfile(png_path):
                pass
            else:
                process_flac_file(file_path, sample_rate, n_mels, hop_length, output_path, x_axis, y_axis, axis)


if __name__ == '__main__':
    args = argparse.ArgumentParser(
            prog='FLAC to MEL spectrogram converter',
            description='Does what it says on the tin according to given parameters')
    
    args.add_argument('--csv', type=str, required=False)
    args.add_argument('--output-dir', type=str, required=False, default='output')
    args.add_argument('--sample-rate', type=int, required=False, default=16000)
    args.add_argument('--n-mels', type=int, required=False, default=64)
    args.add_argument('--hop-length', type=int, required=False, default=160)
    args.add_argument('--delimiter', type=str, required=False, default=' ')
    args.add_argument('--position', type=int, required=False, default=0)
    args.add_argument('--axis', type=str, required=False, default='off')
    args.add_argument('--y-axis', type=str, required=False, default='None')
    args.add_argument('--x-axis', type=str, required=False, default='None')

    if args.parse_args().csv == None:
        args.print_help()
    else:
        args = args.parse_args()
        
        process_flac_files(args.csv, args.delimiter, args.position, args.output_dir, args.sample_rate, args.n_mels,
                           args.hop_length, args.x_axis, args.y_axis, args.axis)
