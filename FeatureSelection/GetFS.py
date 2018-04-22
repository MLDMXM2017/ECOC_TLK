import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from FeatureSelection.FeatureSelector import BSSWSS, VarianceSelector


def get_fs(fs, k):
    fser = None
    if fs == 'bw':
        fser = BSSWSS(k)
    elif fs == 'kf':
        fser = SelectKBest(f_classif, k=k)
    elif fs == 'kmi':
        fser = SelectKBest(mutual_info_classif, k=k)
    elif fs == 'var':
        fser = VarianceSelector(k)
    elif fs == 'chi2':
        fser = SelectKBest(chi2, k=k)
    else:
        raise Exception('Unknown fs.')
    return fser


def fea_rank(X, y, fs):
    fser = get_fs(fs, 1)
    fser.fit(X, y)
    scores = np.nan_to_num(fser.scores_)
    ranking_indices = scores.argsort()[::-1]
    return ranking_indices
