import preprocessing
from classifier import Classifier
from skimage import io
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class Detector(object):
    CHARACTERS = list('abcdefghijklmnopqrstuvwxyz')

    def __init__(self):
        self.classifier = Classifier('extra_trees')
        self.stride = (20, 20)  # (y, x)
        self.window_size = (20, 20)  # (y, x)
        self.threshold = 0.5

    def detect_characters(self, image, verbose=False):
        width = image.shape[1]
        height = image.shape[0]
        print 'width', width
        print 'height', height

        detected_characters = []

        for y in range(0, height, self.stride[1]):
            for x in range(0, width, self.stride[0]):
                window = image[y:y + self.window_size[0], x:x + self.window_size[1]]
                window = preprocessing.Preprocessing.preprocess_image(window)

                probabilities = self.classifier.get_probabilities(window)
                character_probability_pairs = sorted(
                    zip(probabilities, self.CHARACTERS),
                    reverse=True
                )
                if character_probability_pairs[0][0] > self.threshold:
                    if verbose:
                        print
                        print
                        preprocessing.Preprocessing.print_ascii(window)
                        print x, y, 'looks like', character_probability_pairs[0][1]
                    detected_characters.append(
                        (x, y, character_probability_pairs[0][1])
                    )
        return detected_characters

    def visualize_detected_characters(self, image, detected_characters, save_to_filename=None):
        pil_image = Image.fromarray(np.uint8(image))
        pil_image.convert('RGB')
        draw = ImageDraw.Draw(pil_image)

        for roi in detected_characters:
            draw.rectangle(
                [
                    roi[0],
                    roi[1],
                    roi[0] + self.window_size[1],
                    roi[1] + self.window_size[0]
                ],
                outline='#ff0000'
            )
        del draw

        if save_to_filename is not None:
            pil_image.save(save_to_filename)

        return pil_image


if __name__ == '__main__':
    d = Detector()

    path = 'detection-tests'
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.png') and 'result' not in file:
                print file

                file_path = os.path.join(path, file)
                image = io.imread(file_path, as_grey=True, plugin='pil')

                detected_characters = d.detect_characters(image, verbose=True)

                d.visualize_detected_characters(
                    image,
                    detected_characters,
                    save_to_filename=file_path + '.result.png'
                )
