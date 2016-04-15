import os
from sklearn.externals import joblib
import h5py

with h5py.File('data_set.hdf5', 'r') as hf:
    X_tr = hf.get('normalized').get('training').get('default')[:][0]
    Y_tr = hf.get('normalized').get('training').get('targets')[:][0]
    X_va = hf.get('normalized').get('validation').get('default')[:][0]
    Y_va = hf.get('normalized').get('validation').get('targets')[:][0]

USE_EXISTING_CLASSIFIER = False
STORE_CLASSIFIER = True
CLASSIFIER_TYPE = 'nearest_neighbour'  # used for training

classifier_file_path = os.path.join('classifiers', CLASSIFIER_TYPE + '_classifier.pickle')

if USE_EXISTING_CLASSIFIER and os.path.exists(classifier_file_path):
    print 'Using existing classifier'
    classifier = joblib.load(classifier_file_path)
else:
    print 'Training...'
    if CLASSIFIER_TYPE == 'svc':
        from sklearn.svm import SVC

        classifier = SVC(kernel='linear', C=1.0)
    elif CLASSIFIER_TYPE == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier

        classifier = RandomForestClassifier(n_estimators=90)
    elif CLASSIFIER_TYPE == 'nearest_neighbour':
        from sklearn import neighbors

        classifier = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
    else:
        raise Exception('CLASSIFIER variable has an invalid value')

    classifier.fit(X_tr, Y_tr)
    if STORE_CLASSIFIER:
        joblib.dump(classifier, classifier_file_path)

print 'Classifier:', classifier


def test_performance(x, y):
    y_predicted = classifier.predict(x)

    num_predicted = len(y_predicted)
    num_correct_classifications = 0

    for i in range(len(y_predicted)):
        y_predicted[i] = int(round(y_predicted[i]))
        """
        print '======='
        print '  correct classification:', y[i]
        print 'predicted classification:', y_predicted[i]
        """

        if y[i] == y_predicted[i]:
            num_correct_classifications += 1

    print '========================================================='
    int_format = '{0: >50}: {1}'
    percentage_format = '{0: >50}: {1:.1f} %'

    print int_format.format('Total number of items classified', num_predicted)
    print percentage_format.format(
            'Correct classifications (percentage)',
            100 * float(num_correct_classifications) / num_predicted
    )

print 'Testing performance on validation set...'
test_performance(X_va, Y_va)
print
# print 'Testing performance on training set...'
# test_performance(X_tr, Y_tr)
