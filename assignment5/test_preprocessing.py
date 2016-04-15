import unittest
import preprocessing
from skimage import io
import os


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_one_image(self):
        p = preprocessing.Preprocessing()
        p.load_files()

        for i in [0, 500, 900, 2000, 3000, 4000, 5000, 6000, 7000]:
            image = p.x[i]
            io.imsave(os.path.join('tmp', 'image{}.png'.format(i)), image)

            processed_image = p.preprocess_image(image)
            io.imsave(os.path.join('tmp', 'image{}_after.png'.format(i)), processed_image)

            variations = p.create_variations(processed_image, character='o')
            for j in range(len(variations)):
                io.imsave(
                    os.path.join('tmp', 'image{0}_after_variation{1}.png'.format(i, j)),
                    variations[j]
                )


if __name__ == '__main__':
    unittest.main()
