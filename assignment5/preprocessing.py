#!/usr/bin/env python
# coding=utf-8
from PIL import Image
import os
import h5py
import numpy as np
import random
from skimage import io, filters, feature, transform, exposure
from skimage.restoration import denoise_bilateral


class Preprocessing(object):
    def __init__(self):
        self.x, self.y = None, None
        self.training_inputs = None
        self.training_targets = None
        self.validation_inputs = None
        self.validation_targets = None
        self.num_entries = None
        self.num_training_entries = None
        self.num_validation_entries = None
        self.vector_size = None

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

    def load_files(self):
        self.x = []
        self.y = []
        prefix = 'chars74k-lite'
        characters = 'abcdefghijklmnopqrstuvwxyz'

        for label_index in range(len(characters)):
            path = os.path.join(prefix, characters[label_index])
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith('.jpg'):
                        # print file

                        file_path = os.path.join(path, file)
                        image = io.imread(file_path, as_grey=True, plugin='pil')
                        self.x.append(image)
                        self.y.append(label_index)

    def divide_data_set(self, shuffle_seed=42):
        # shuffle the data set to have all characters in both sets and to prevent positional bias
        zipped_data_set = zip(self.x, self.y)
        random.seed(shuffle_seed)
        random.shuffle(zipped_data_set)
        self.x = [x[0] for x in zipped_data_set]
        self.y = [x[1] for x in zipped_data_set]

        self.num_entries = len(self.x)
        self.num_training_entries = int(self.num_entries * 0.7)
        self.num_validation_entries = self.num_entries - self.num_training_entries
        self.vector_size = self.x[0].size

        self.training_inputs = self.x[:self.num_training_entries]
        self.training_targets = self.y[:self.num_training_entries]
        self.validation_inputs = self.x[self.num_training_entries:]
        self.validation_targets = self.y[self.num_training_entries:]

    def augment_training_set(self):
        angles = [5, 10, -5, -10]

        for angle in angles:
            for i in range(len(self.training_inputs)):
                image = self.training_inputs[i]
                label = self.training_targets[i]
                transformed_image = transform.rotate(
                    image,
                    angle,
                    resize=False,
                    preserve_range=True
                )
                self.training_inputs.append(transformed_image)
                self.training_targets.append(label)
                self.num_training_entries += 1
                self.num_entries += 1

    def write_data_set(self):
        print('num_entries', self.num_entries)
        print('num_training_entries', self.num_training_entries)
        print('num_validation_entries', self.num_validation_entries)
        print('input vector size', self.vector_size)

        file_name = 'data_set.hdf5'
        hdf_file_path = os.path.join('.', file_name)
        f = h5py.File(hdf_file_path, 'w')

        variant = f.create_group('normalized')

        training_group = variant.create_group('training')
        training_group.create_dataset(
            name='default',
            data=np.array(self.training_inputs).reshape(
                (1, self.num_training_entries, self.vector_size)
            ),
            compression='gzip'
        )
        training_group.create_dataset(
            name='targets',
            data=np.array(self.training_targets).reshape(
                (1, self.num_training_entries)
            ),
            compression='gzip'
        )

        validation_group = variant.create_group('validation')
        validation_group.create_dataset(
            name='default',
            data=np.array(self.validation_inputs).reshape(
                (1, self.num_validation_entries, self.vector_size)
            ),
            compression='gzip'
        )
        validation_group.create_dataset(
            name='targets',
            data=np.array(self.validation_targets).reshape(
                (1, self.num_validation_entries)
            ),
            compression='gzip'
        )

        f.close()

    def preprocess_images(self):
        for i in range(len(self.x)):
            self.x[i] = self.preprocess_image(self.x[i])

    @staticmethod
    def preprocess_image(image):
        # image = filters.sobel(image)  # sobel doesn't seem to improve predictive performance
        # image = denoise_bilateral(image, sigma_range=0.05, sigma_spatial=4, multichannel=False)  # computationally expensive
        # image = feature.canny(image)  # outputs values between 0 and 1

        p2, p98 = np.percentile(image, (2, 98))
        image = exposure.rescale_intensity(image, in_range=(p2, p98))

        return image


if __name__ == '__main__':
    p = Preprocessing()
    p.load_files()
    p.preprocess_images()
    p.divide_data_set()
    p.augment_training_set()
    p.write_data_set()
