
""" Base Classifiers """


def get_base_clf(base, adaboost=False):
    """ Get classifiers from scikit-learn.

    Parameters:
        base: str
            indicates classifier, alternative str list below.
            'KNN' - K Nearest Neighbors (sklearn.neighbors.KNeighborsClassifier)
            'DTree' - Decision Tree (sklearn.tree.DecisionTreeClassifier)
            'SVM' - Support Vector Machine (sklearn.svm.SVC)
            'Bayes' - Naive Bayes (sklearn.naive_bayes.GaussianNB)
            'Logi' - Logistic Regression (sklearn.linear_model.LogisticRegression)
            'NN' - Neural Network (sklearn.neural_network.MLPClassifier)
        adaboost: bool, default False.
            Whether to use adaboost to promote the classifier.

    Return:
        model: object, A classifier object.
    """
    model = None
    if base is 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        adaboost = False
    elif base is 'DTree':
        from sklearn import tree
        model = tree.DecisionTreeClassifier()
    elif base is 'SVM':
        from sklearn.svm import SVC
        model = SVC()
        adaboost = False
    elif base is 'Bayes':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        adaboost = False
    elif base is 'Logi':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif base is 'NN':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier()
        adaboost = False
    else:
        raise ValueError('Classify: Unknown value for base: %s' % base)

    # if use an adabost to strengthen model.
    if adaboost is True:
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(model, algorithm="SAMME")

    return model
