import numpy as np

# CONSTANTS
DTYPE = np.float64

INPUT_SPACE_DIM = 3  # dimension of the input space
LAT_RANGE = np.array([-180, 180])  # latitude range
LON_RANGE = np.array([-90, 90])  # longitude range
DATE_RANGE = np.array([1960, 2013])  # date range

RADII_RANGE = np.array([0, 1])  # radius range for the bump functions
AMPL_RANGE = np.array([0, 10])  # amplitude range for the bump functions
SHAPE_RANGE = np.array([0, 10])  # shape parameter range for the bump functions

R_THRESHOLD_RANGE = np.array([0, 100])  # threshold for the k_tilde function
LENGTH_SCALE_RANGE = np.array([0, 100])  # length scale range
PERIOD_RANGE = np.array([0, 100])  # period for the periodic kernel

X_BOUNDS = np.array([LAT_RANGE, LON_RANGE, DATE_RANGE])  # bounds for the  input space


def build_hps(n1, n2, D=3):
    '''
    Build the hyperparameters for the sparse-discovering kernel
    combined with a periodic kernel.
    Parameters:
    ----------
    n1: int
        "hyper-hyperparameter" for the sparse-discovering kernel, is the n1 in formula (10) of the considered paper.
    n2: int
        "hyper-hyperparameter" for the sparse-discovering kernel, is the n2 in formula (10) of  the considered paper.
    D: int
        dimension of the input space, default (for our case) is 3.
    '''

    hps = []

    # Bump functions hyperparameters
    for i in range(n1):
        for j in range(n2):
            for d in range(D):
                hps.append(np.random.uniform(*X_BOUNDS[d]))
    # Radius for the bump functions
    for j in range(n2):
        hps.append(np.random.uniform(*RADII_RANGE))

    # Amplitudes for the bump functions
    for j in range(n2):
        hps.append(np.random.uniform(*AMPL_RANGE))

    # Shape parameter for the bump functions
    for j in range(n2):
        hps.append(np.random.uniform(*SHAPE_RANGE))

    # Signal variance, threshold for the chi function and length scale for the bump functions
    hps.append(np.random.uniform(*R_THRESHOLD_RANGE))
    hps.append(np.random.uniform(*LENGTH_SCALE_RANGE))
    hps.append(np.random.uniform(*PERIOD_RANGE))

    return np.array(hps, dtype=DTYPE)


def build_bounds(n1, n2, D=3):
    '''
    Build the bounds for the hyperparameters for the sparse-discovering kernel
    combined with a periodic kernel.
    Parameters:
    ----------
    n1: int
        "hyper-hyperparameter" for the sparse-discovering kernel, is the n1 in formula (10) of the considered paper.
    n2: int
        "hyper-hyperparameter" for the sparse-discovering kernel, is the n2 in formula (10) of the considered paper.
    D: int
        dimension of the input space, default (for our case) is 3.
    '''

    bounds = []

    # Bump functions hyperparameters
    for i in range(n1):
        for j in range(n2):
            for d in range(D):
                bounds.append(X_BOUNDS[d])
    # Radius for the bump functions
    for j in range(n2):
        bounds.append(RADII_RANGE)

    # Amplitudes for the bump functions
    for j in range(n2):
        bounds.append(AMPL_RANGE)

    # Shape parameter for the bump functions
    for j in range(n2):
        bounds.append(SHAPE_RANGE)

    # Signal variance, threshold for the chi function and length scale for the bump functions
    bounds.append(R_THRESHOLD_RANGE)
    bounds.append(LENGTH_SCALE_RANGE)
    bounds.append(PERIOD_RANGE)

    return np.array(bounds, dtype=DTYPE)


def count_hps(n1, n2, D=3):
    '''
    Count the number of hyperparameters for the sparse-discovering kernel
    combined with a periodic kernel.
    Parameters:
    ----------
    n1: int
        "hyper-hyperparameter" for the sparse-discovering kernel, is the n1 in formula (10) of the considered paper.
    n2: int
        "hyper-hyperparameter" for the sparse-discovering kernel, is the n2 in formula (10) of the considered paper.
    D: int
        dimension of the input space, default (for our case) is 3.
    '''
    return D * n1 * n2 + 3 * n2 + 3
