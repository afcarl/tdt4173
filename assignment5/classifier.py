import os
from sklearn.externals import joblib
from preprocessing import Preprocessing
import cPickle


class Classifier(object):
    def __init__(self, classifier_type='extra_trees'):
        classifier_file_path = os.path.join(
            'classifiers',
            classifier_type + '_classifier.pickle'
        )

        if classifier_type in ['random_forest', 'extra_trees']:
            with open(classifier_file_path, 'rb') as f:
                self.classifier = cPickle.load(f)
        else:
            self.classifier = joblib.load(classifier_file_path)

    @staticmethod
    def get_category_by_index(idx):
        return 'abcdefghijklmnopqrstuvwxyz'[idx]

    def classify(self, image):
        processed_image = Preprocessing.preprocess_image(image)
        input_vector = processed_image.flatten()
        category_vector = self.classifier.predict([input_vector])
        category_index = category_vector[0]
        category = self.get_category_by_index(category_index)
        return category

    def get_probabilities(self, image):
        processed_image = Preprocessing.preprocess_image(image)
        input_vector = processed_image.flatten()
        category_vector = self.classifier.predict_proba([input_vector])
        return category_vector[0]
