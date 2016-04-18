import preprocessing
from classifier import Classifier
from skimage import io
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class Detector(object):
    CHARACTERS = list('abcdefghijklmnopqrstuvwxyz')

    def __init__(self, classifier_type):
        self.classifier = Classifier(classifier_type)
        self.stride = (20, 20)  # (y, x)
        self.window_size = (20, 20)  # (y, x)
        self.threshold = 0.5

    def detect_characters(self, image, verbose=False):
        """
        Apply the rolling window technique on the given image
        Returns a list of tuples like (x, y, character)
        """

        width = image.shape[1]
        height = image.shape[0]
        print 'width', width
        print 'height', height

        detected_characters = []

        for y in range(0, height - self.stride[1] + 1, self.stride[1]):
            for x in range(0, width - self.stride[0] + 1, self.stride[0]):
                window = image[y:y + self.window_size[0], x:x + self.window_size[1]]
                if window.shape != (20, 20):
                    continue
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
        """
        Take an image and red draw rectangles and detected characters on the given positions
        The image is stored if save_to_filename is specified
        """
        base_pil_image = Image.fromarray(np.uint8(image)).convert('RGBA')
        new_pil_image = Image.new('RGBA', base_pil_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(new_pil_image)

        font = ImageFont.truetype('theboldfont.ttf', 22)
        font_offset_x = 3
        font_offset_y = 2

        for roi in detected_characters:
            draw.rectangle(
                [
                    roi[0],
                    roi[1],
                    roi[0] + self.window_size[1],
                    roi[1] + self.window_size[0]
                ],
                fill=(190, 190, 190, 190),
                outline=(220, 40, 40, 255)
            )

            draw.text(
                (roi[0] + font_offset_x, roi[1] + font_offset_y),
                roi[2],
                font=font,
                fill=(255, 0, 0, 255)
            )

        del draw

        result = Image.alpha_composite(base_pil_image, new_pil_image)

        if save_to_filename is not None:
            result.save(save_to_filename)

        return result


if __name__ == '__main__':
    # Run detection
    d = Detector('extra_trees')

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
