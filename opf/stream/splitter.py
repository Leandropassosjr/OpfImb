import numpy as np

import opf.utils.exception as e
import opf.utils.logging as l

logger = l.get_logger(__name__)


# def k_folds(X, Y, n_folds=2, random_state=1):
#     """Splits the data into k-folds for further cross-validation.

#     Args:
#         X (np.array): Array of features.
#         Y (np.array): Array of labels.
#         n_folds (int): Number of folds (`k` value).
#         random_state (int): An integer that fixes the random seed.

#     Returns:
#         k-folds that were created from `X` and `Y`.

#     """

#     logger.info(f'Creating k-folds with k = {n_folds} ...')

#     logger.info('Folds created.')



#def separate(X,idx, idx1, idx2, n_elements):
#	test = X[idx[idx1:idx2], ...]
#	train = X[idx[-(n_elements-idx2):idx1],...]
#	return train, test

#def k_fold(X, Y, index, n_folds=5, random_state=1):
#    """Splits data into k-folds.

#    Args:
#        X (np.array): Array of features.
#        Y (np.array): Array of labels.
#		index (np.array): Array of indexes.
#        n_folds (int): Number of train/test sets generated.
#        random_state (int): An integer that fixes the random seed.
#
#    Returns:
#        Two new sets that were created from `X` and `Y`.
#    """
#
#    logger.info(f'Splitting data ...')
#
#    # Defining a fixed random seed
#    np.random.seed(random_state)
#
#    # Checks if `X` and `Y` have the same size
#    if X.shape[0] != Y.shape[0]:
#        # If not, raises a SizeError
#        raise e.SizeError(
#            f'`X` and `Y` should have the same amount of samples')
#    # Checks if `X` and `index` have the same size
#    if X.shape[0] != index.shape[0]:
#        # If not, raises a SizeError
#        raise e.SizeError(
#            f'`X` and `index` should have the same amount of samples')
#
#    # Gathering the indexes
#    idx = np.random.permutation(X.shape[0])

#    Xs_train = []
#    Ys_train = []
#    indexes_train = []

#    Xs_test = []
#    Ys_test = []
#    indexes_test = []
#    n_by_fold = int(len(X)/n_folds)
#    for i in range(n_folds):
#        idx1 = n_by_fold * i
#        idx2 = n_by_fold * (i+1)

		# Gathering two new sets from `X`
#		train, test = separate(X,idx,idx1, idx2, len(X)):
#        Xs_traint.append(train)
#        Xs_test.append( test)

#		# Gathering two new sets from `Y`
#		train, test = separate(Y,idx,idx1, idx2, len(X)):
#        Ys_traint.append(train)
#        Ys_test.append( test)
#        #Ys_test.append( Y[idx[idx1:idx2]])

#		# Gathering two new sets from `index`
#		train, test = separate(index,idx,idx1, idx2, len(X)):
#        indexes_traint.append(train)
#        indexes_test.append( test)
#        #indexes_test.append( index[idx1:idx2])


#    Xs = np.asarray(Xs)
#    Ys = np.asarray(Ys)
#    indexes = np.asarray(indexes)

#    logger.debug(
#        f'Xs: {Xs.shape} | Ys: {Ys.shape} | indexes: {indexes.shape} .')
#    logger.info('k-fold spplited.')


#    Xs = np.asarray(Xs)
#    Ys = np.asarray(Ys)
#    indexes = np.asarray(indexes)

#    return Xs, Ys, indexes


def split(X, Y, index, percentage=0.5, random_state=1):
    """Splits data into two new sets.

    Args:
        X (np.array): Array of features.
        Y (np.array): Array of labels.
		index (np.array): Array of indexes.
        percentage (float): Percentage of the data that should be in first set.
        random_state (int): An integer that fixes the random seed.

    Returns:
        Two new sets that were created from `X` and `Y`.

    """

    logger.info(f'Splitting data ...')

    # Defining a fixed random seed
    np.random.seed(random_state)

    # Checks if `X` and `Y` have the same size
    if X.shape[0] != Y.shape[0]:
        # If not, raises a SizeError
        raise e.SizeError(
            f'`X` and `Y` should have the same amount of samples')

    # Gathering the indexes
    idx = np.random.permutation(X.shape[0])

    # Calculating where sets should be halted
    halt = int(len(X) * percentage)

    # Gathering two new sets from `X`
    X_1, X_2 = X[idx[:halt], :], X[idx[halt:], :]

    # Gathering two new sets from `Y`
    Y_1, Y_2 = Y[idx[:halt]], Y[idx[halt:]]

    # Gathering two new sets from `index`
    index_1, index_2 = index[idx[:halt]], index[idx[halt:]]

    logger.debug(
        f'X_1: {X_1.shape} | X_2: {X_2.shape} | Y_1: {Y_1.shape} | Y_2: {Y_2.shape} | index_1: {index_1.shape} | index_2: {index_2.shape}.')
    logger.info('Data splitted.')

    return X_1, X_2, Y_1, Y_2, index_1, index_2


def merge(X_1, X_2, Y_1, Y_2):
    """Merge two sets into a new set.

    Args:
        X_1 (np.array): First array of features.
        X_2 (np.array): Second array of features.
        Y_1 (np.array): First array of labels.
        Y_2 (np.array): Second array of labels.

    Returns:
        A new merged set that was created from `X_1`, `X_2`, `Y_1` and `Y_2`.

    """

    logger.info(f'Merging data ...')

    # Vertically stacking `X_1` and `X_2`
    X = np.vstack((X_1, X_2))

    # Horizontally stacking `Y_1` and Y_2`
    Y = np.hstack((Y_1, Y_2))

    # Checks if `X` and `Y` have the same size
    if X.shape[0] != Y.shape[0]:
        # If not, raises a SizeError
        raise e.SizeError(
            f'`(X_1, X_2)` and `(Y_1, Y_2)` should have the same amount of samples')

    logger.debug(f'X: {X.shape} | Y: {Y.shape}.')
    logger.info('Data merged.')

    return X, Y
