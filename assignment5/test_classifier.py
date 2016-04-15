import unittest
import preprocessing
from classifier import Classifier


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.p = preprocessing.Preprocessing()
        self.p.load_files()
        self.c = Classifier('nearest_neighbour')

    def test_classifier(self):
        alphabet = list('abcdefghijklmnopqrstuvwxyz')

        for i in range(0, len(self.p.x), 100):
            image = self.p.x[i]
            preprocessing.Preprocessing.print_ascii(image)
            print 'predicted to be', self.c.classify(image)
            probabilities = self.c.get_probabilities(image)
            character_probability_pairs = sorted(zip(probabilities, alphabet), reverse=True)
            most_probable_characters = [
                '{0}: {1:.2f}'.format(x[1], x[0]) for x in character_probability_pairs[0:3]
                if x[0] > 0.0
                ]
            print ', '.join(most_probable_characters)
            print

if __name__ == '__main__':
    unittest.main()
