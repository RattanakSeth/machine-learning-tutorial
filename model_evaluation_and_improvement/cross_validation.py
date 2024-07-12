from sklearn.datasets import make_blobs, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import TimeSeriesSplit
import numpy as np;


def train_test_split():
    # create a synthetic dataset
    X, y = make_blobs(random_state=0)
    # split data and labels into a training and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # instantiate a model and fit it to the training set
    logreg = LogisticRegression().fit(X_train, y_train)
    # evaluate the model on the test set
    print("Test set score: {:.2f}".format(logreg.score(X_test, y_test)))

    # print
    print("X_tain:", X_train)
    print("y_test label as a vector: ", y_test)

#    .. versionchanged:: 0.22
#             `cv` default value if `None` changed from 3-fold to 5-fold.
def crossValidationInScikitLearn():
    iris = load_iris()
    logreg = LogisticRegression()
    scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {:.2f}".format(scores.mean()))

   
"""K-Folds cross-validator

Provides train/test indices to split data in train/test sets. Split
dataset into k consecutive folds (without shuffling by default).

Each fold is then used once as a validation while the k - 1 remaining
folds form the training set.

Read more in the :ref:`User Guide <k_fold>`.

Parameters
----------
n_splits : int, default=5
    Number of folds. Must be at least 2.

    .. versionchanged:: 0.22
        ``n_splits`` default value changed from 3 to 5.

shuffle : bool, default=False
    Whether to shuffle the data before splitting into batches.
    Note that the samples within each split will not be shuffled.

random_state : int, RandomState instance or None, default=None
    When `shuffle` is True, `random_state` affects the ordering of the
    indices, which controls the randomness of each fold. Otherwise, this
    parameter has no effect.
    Pass an int for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import KFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([1, 2, 3, 4])
>>> kf = KFold(n_splits=2)
>>> kf.get_n_splits(X)
2
>>> print(kf)
KFold(n_splits=2, random_state=None, shuffle=False)
>>> for i, (train_index, test_index) in enumerate(kf.split(X)):
...     print(f"Fold {i}:")
...     print(f"  Train: index={train_index}")
...     print(f"  Test:  index={test_index}")
Fold 0:
    Train: index=[2 3]
    Test:  index=[0 1]
Fold 1:
    Train: index=[0 1]
    Test:  index=[2 3]

Notes
-----
The first ``n_samples % n_splits`` folds have size
``n_samples // n_splits + 1``, other folds have size
``n_samples // n_splits``, where ``n_samples`` is the number of samples.

Randomized CV splitters may return different results for each call of
split. You can make the results identical by setting `random_state`
to an integer.

See Also
--------
StratifiedKFold : Takes class information into account to avoid building
    folds with imbalanced class distributions (for binary or multiclass
    classification tasks).

GroupKFold : K-fold iterator variant with non-overlapping groups.

RepeatedKFold : Repeats K-Fold n times.
"""
def moreControlOverCrossValidation():
    iris = load_iris()
    print("Iris labels:\n{}".format(iris.target))
    logreg = LogisticRegression()
    kfold = KFold(n_splits=5)
    print("Kfold: ", kfold)

    # Then, we can pass the kfold splitter object as the cv parameter to cross_val_score:
    print("Cross-validation scores:\n{}".format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)))


# TimeSeriesSplit is a variation of k-fold which returns first 
#  folds as train set and the 
#  th fold as test set. Note that unlike standard cross-validation methods, successive training sets are supersets of those that come before them. Also, it adds all surplus data to the first training partition, which is always used to train the model.
# This class can be used to cross-validate time series data samples that are observed at fixed time intervals.
# Example of 3-split time series cross-validation on a dataset with 6 samples:
def timeSeriesSplit():
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4, 5, 6])
    tscv = TimeSeriesSplit(n_splits=2)
    print(tscv.split(y)) 
    # TimeSeriesSplit(gap=0, max_train_size=None, n_splits=3, test_size=None)
    
    for train, test in tscv.split(y):
        print("%s %s" % (train, test))


#crossValidationInScikitLearn()
moreControlOverCrossValidation()
# timeSeriesSplit()