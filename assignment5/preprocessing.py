#!/usr/bin/env python
# coding=utf-8
from PIL import Image
import os
import h5py
import numpy as np
import random


class Preprocessing(object):
    def __init__(self):
        self.x, self.y = self.get_vectors()
        self.shuffle_data_set()
        self.write_data_set()

    @staticmethod
    def print_ascii(image):
        chars = u' ░▒▓█'
        width, height = image.size
        pixel_values = list(image.getdata())
        for y in range(height):
            r = ''
            for x in range(width):
                pixel_value = float(pixel_values[width * y + x]) / 255
                char_idx = int(round(pixel_value * (len(chars) - 1)))
                r += chars[char_idx] + chars[char_idx]
            print r

    @staticmethod
    def get_vectors():
        x = []
        y = []
        prefix = 'chars74k-lite'
        characters = 'abcdefghijklmnopqrstuvwxyz'

        for label_index in range(len(characters)):
            path = os.path.join(prefix, characters[label_index])
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith('.jpg'):
                        # print file

                        file_path = os.path.join(path, file)
                        image = Image.open(file_path, 'r')
                        pixel_values = list(image.getdata())
                        x.append(pixel_values)
                        y.append(label_index)

        return x, y

    def shuffle_data_set(self):
        # shuffle the data set to have all characters in both sets and to prevent positional bias
        zipped_data_set = zip(self.x, self.y)
        random.seed(42)
        random.shuffle(zipped_data_set)
        self.x = [x[0] for x in zipped_data_set]
        self.y = [x[1] for x in zipped_data_set]

    def write_data_set(self):
        file_name = 'data_set.hdf5'
        hdf_file_path = os.path.join('.', file_name)
        f = h5py.File(hdf_file_path, 'w')

        num_entries = len(self.x)
        num_training_entries = int(num_entries * 0.7)
        num_validation_entries = num_entries - num_training_entries
        vector_size = len(self.x[0])

        print('num_entries', num_entries)
        print('num_training_entries', num_training_entries)
        print('num_validation_entries', num_validation_entries)
        print('input vector size', vector_size)

        training_inputs = np.array(self.x[:num_training_entries]).reshape(
            (1, num_training_entries, vector_size)
        )
        training_targets = np.array(self.y[:num_training_entries]).reshape(
            (1, num_training_entries)
        )
        validation_inputs = np.array(self.x[num_training_entries:]).reshape(
            (1, num_validation_entries, vector_size)
        )
        validation_targets = np.array(self.y[num_training_entries:]).reshape(
            (1, num_validation_entries)
        )

        all_inputs = np.array(self.x[:]).reshape(
            (1, num_entries, vector_size)
        )
        all_targets = np.array(self.y[:]).reshape(
            (1, num_entries)
        )

        variant = f.create_group('normalized')

        training_group = variant.create_group('training')
        training_group.create_dataset(name='default', data=training_inputs, compression='gzip')
        training_group.create_dataset(name='targets', data=training_targets, compression='gzip')

        validation_group = variant.create_group('validation')
        validation_group.create_dataset(name='default', data=validation_inputs, compression='gzip')
        validation_group.create_dataset(name='targets', data=validation_targets, compression='gzip')

        all_group = variant.create_group('all')
        all_group.create_dataset(name='default', data=all_inputs, compression='gzip')
        all_group.create_dataset(name='targets', data=all_targets, compression='gzip')

        f.close()


if __name__ == '__main__':
    Preprocessing()
