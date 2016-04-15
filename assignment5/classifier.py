import os
from sklearn.externals import joblib
from preprocessing import Preprocessing


class Classifier(object):
    def __init__(self, classifier_type='random_forest'):
        classifier_file_path = os.path.join(
            'classifiers',
            classifier_type + '_classifier.pickle'
        )
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
