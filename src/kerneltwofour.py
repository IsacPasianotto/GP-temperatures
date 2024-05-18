# imports
import numpy as np

# CONSTANTS
INPUT_SPACE_DIM = 3  # dimension of the input space
N1 = 2
N2 = 4

def l2_norm(x1, x2):
    '''
    Compute the L2 norm between two vectors. This is the considered distance function for the kernel.
    Parameters:
    ----------
    x1: np.array
        first vector
    x2: np.array
        second vector
    '''
    return np.sqrt(np.sum((x1 - x2)**2))


def custom_kernel(x1, x2, hyperparameters, obj):
    '''
    Custom kernel for the GP class. It is a combination of a sparse-discovering kernel and a periodic kernel.
    Parameters:
    ----------
    x1: np.array
        first vector
    x2: np.array
        second vector
    hyperparameters: np.array
        hyperparameters for the kernel. Should be a (n1*n2*D + 2*n2 + 3) array, with hyperparameters ordered as follows:
            - n1*n2*D  ***: coordinates for the bump functions centers      (x0_ij in the paper)
            - n2  ***: radii for the bump functions                         (r_ij in the paper)
            - n2  ***: amplitudes for the bump functions                    (alpha_ij in the paper)
            - n2  ***: shape parameter for the bump functions               (beta_ij in the paper)
            - 1  ***: threshold for the k_tilde function                    (r in the paper)
            - 1  ***: length scale for the K_s                              (novelty respect to the paper: needed for the k_p kernel)
            - 1  ***: period for the periodic kernel K_p                    (novelty respect to the paper: needed for the k_p kernel)
    obj: GP
        GP object
    '''
    # For now, we need to hard-code in the function the values of n1, n2 and D.
    # Make it parametric requires to re-implement the library!
    n1 = N1
    n2 = N2
    D = INPUT_SPACE_DIM  # 3

    ## Extract the hyperparameters ##

    centers = hyperparameters[:n1 * n2 * D].reshape(n1, n2, D)  # Bump functions centers --> n1*n2 values in R^D: [[x0_11, x0_12, x0_13], [x0_21, x0_22, x0_23],...]
    # centers = hyperparameters[:n1*n2*D]                       # Bump functions centers --> n1*n2 values: [x0_11, x0_12, x0_13, x0_21, x0_22, x0_23,...]
    radii = hyperparameters[n1 * n2 * D:n1 * n2 * D + n2]
    amplitudes = hyperparameters[n1 * n2 * D + n2:n1 * n2 * D + 2 * n2]
    shape_parameters = hyperparameters[n1 * n2 * D + 2 * n2:n1 * n2 * D + 3 * n2]

    r = hyperparameters[-3]

    length_scale = hyperparameters[-2]
    period = hyperparameters[-1]

    ## Compute the K_s kernel ##

    def l2_norm(x1, x2):
        '''
        Compute the L2 norm between two vectors. This is the considered distance function for the kernel.
        Parameters:
        ----------
        x1: np.array
            first vector
        x2: np.array
            second vector
        '''
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def f_i(x, i, j):
        '''
        Compute the bump function f_i(x) for the sparse-discovering kernel.
        Parameters:
        ----------
        x: np.array
            input vector
        i: int
            index of the bump function, i in {1, ..., n1}
        j: int
            index of the bump function, j in {1, ..., n2}
        '''
        x0 = centers[i, j]
        r = radii[j]
        a = amplitudes[j]
        beta = shape_parameters[j]
        chi = 0
        if r < l2_norm(x, x0):
            chi = 1
        return a * np.exp(-1 * beta / (1 - (l2_norm(x, x0) ** 2) / r ** 2)) * chi

    def tilde_k(x1, x2, r):
        '''
        Compute the tilde_k kernel for the sparse-discovering kernel.
        Parameters:
        ----------
        x1: np.array
            first vector
        x2: np.array
            second vector
        r: float
            threshold for the k_tilde function
        '''
        d = l2_norm(x1, x2)
        if d == 0:
            d = 1e-10  # avoid log(0)
        if d < r:
            return np.sqrt(2) / (np.pi ** (1 / 3)) * (
                        3 * (d / r) ** 2 * np.log((d / r) / (1 + np.sqrt(1 - (d / r) ** 2))) + (
                            2 * (d / r) ** 2 + 1) * np.sqrt(1 - (d / r) ** 2))
        return 0

    def k_s(x1, x2):
        '''
        Compute the K_s kernel for the sparse-discovering kernel.
        Parameters:
        ----------
        x1: np.array
            first vector
        x2: np.array
            second vector
        '''
        return tilde_k(x1, x2, r) * np.sum([f_i(x1, i, j) * f_i(x2, i, j) for i in range(n1) for j in range(n2)])

    ## Compute the K_p kernel ##

    def k_p(x1, x2):
        '''
        Compute the K_p kernel for the periodic kernel.
        Parameters:
        ----------
        x1: np.array
            first vector
        x2: np.array
            second vector
        '''
        return np.exp(-2 * np.sin(np.pi * l2_norm(x1, x2) / period) ** 2 / length_scale ** 2)

    ## Compute the final kernel ##

    return k_s(x1, x2) * k_p(x1, x2)

def custom_kernel_matrix(x_data, x_data1, hyperparameters, obj):
    N = x_data.shape[0]
    NN = x_data1.shape[0]
    K = np.empty([N, NN])
    # The matrix is symmetric, so we can compute only the upper triangular part
    for i in range(N):
        for j in range(i, NN):
            K[i, j] = custom_kernel(x_data[i], x_data1[j], hyperparameters, obj)
            K[j, i] = K[i, j]
    return K




#### ONE_FUNCTION_FOR_ALL ####

# Since in python the funcion call is expensive, we define a single function that computes the kernel and the kernel matrix
# the code above is the same and more readable

def custom_kernel_one_shot(x_data, x_data1, hyperparameters, obj):
    """
    Compute the kernel matrix for the custom kernel. This function minimizes the overhead of the function calls
    Parameters:
    ---
    x_data: np.array
        first set of data points
    x_data1: np.array
        second set of data points
    hyperparameters: np.array
        hyperparameters for the kernel
    obj: GP
        GP object
    """

    N = x_data.shape[0]
    NN = x_data1.shape[0]
    K = np.empty([N, NN])
    # The matrix is symmetric, so we can compute only the upper triangular part
    for i in range(N):
        for j in range(i, NN):

            ## extract the hyperparameters and constants ##
            x1 = x_data[i]
            x2 = x_data1[j]
            n1 = N1
            n2 = N2
            D = INPUT_SPACE_DIM
            SQRT2 = np.sqrt(2)          # no need to compute it every time

            centers = hyperparameters[:n1 * n2 * D].reshape(n1, n2, D)
            radii = hyperparameters[n1 * n2 * D:n1 * n2 * D + n2]
            amplitudes = hyperparameters[n1 * n2 * D + n2:n1 * n2 * D + 2 * n2]
            shape_parameters = hyperparameters[n1 * n2 * D + 2 * n2:n1 * n2 * D + 3 * n2]
            r = hyperparameters[-3]
            length_scale = hyperparameters[-2]
            period = hyperparameters[-1]

            ## Compute the K_p kernel ##

            K_period = np.exp(-2 * np.sin(np.pi * l2_norm(x1, x2) / period) ** 2 / length_scale ** 2)

            ############################
            ## Compute the K_s kernel ##
            ############################

            ## Compute the tilde_k kernel ##

            k_tilde = 0
            d = np.sqrt(np.sum((x1 - x2)**2))  # l2_norm(x1, x2)
            if d == 0:
                d = 1e-10                      # avoid log(0)
            if d < r:
                k_tilde = SQRT2 / (np.pi ** (1 / 3)) * (
                            3 * (d / r) ** 2 * np.log((d / r) / (1 + np.sqrt(1 - (d / r) ** 2))) + (
                                2 * (d / r) ** 2 + 1) * np.sqrt(1 - (d / r) ** 2))

            ## Compute the sum of bump functions ##

            k_sparse = 0

            for ii in range(n1):
                for jj in range(n2):
                    x0 = centers[ii, jj]
                    r = radii[jj]
                    a = amplitudes[jj]
                    beta = shape_parameters[jj]
                    chi = 0
                    if r < np.sqrt(np.sum((x1 - x0)**2)):
                        chi = 1
                    k_sparse += a * np.exp(-1 * beta / (1 - np.sum((x1 - x0)**2) / r ** 2)) * chi * \
                                a * np.exp(-1 * beta / (1 - np.sum((x2 - x0)**2) / r ** 2)) * chi

            ### Compute the final kernel ###

            K[i, j] = k_tilde * k_sparse * K_period
            K[j, i] = K[i, j]

    # Finally, return the kernel matrix

    return K
