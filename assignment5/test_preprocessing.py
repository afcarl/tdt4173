import unittest
import preprocessing
from skimage import io


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_one_image(self):
        p = preprocessing.Preprocessing()
        p.load_files()

        for i in [0, 500, 900, 2000]:
            image = p.x[i]
            io.imsave('image{}.png'.format(i), image)

            processed_image = p.preprocess_image(image)
            io.imsave('image{}_after.png'.format(i), processed_image)


if __name__ == '__main__':
    unittest.main()
