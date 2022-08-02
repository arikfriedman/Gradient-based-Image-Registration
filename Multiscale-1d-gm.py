import numpy as np
from scipy.ndimage import gaussian_filter1d
import cv2
import sys
from scipy import ndimage


# function which interpolates the signal I by dx:
def interpolate_by_dx(I, dx):
    full_shifted = np.zeros(len(I))
    int_shifted = ndimage.shift(I, np.floor(dx))  # first, shift by the nearest integer
    int_shifted = np.append(int_shifted, 0)
    remainder = dx - np.floor(dx)  # between 0 and 1
    # after shifting by integer, we interpolate with the remaining dx:
    for index in range(int(np.floor(dx)), len(I)):
        full_shifted[index] = \
            int_shifted[index] * (1-remainder) + int_shifted[index-1] * remainder
    # return the cropped interpolated signal:
    return full_shifted[int(np.ceil(dx)):len(I)]


'''This following function performs the same functionality as the original "iterative-1d-gm", but with some initial
dx (which can be different from 0).
We know that I2(xi) = I1(xi +dx) for some dx, which we would like to estimate.
We would now want to create a system of linear equations which we will solve using LS (least squares).
The system of equations is of the following format: Ax = y, where "A" (in our case is) vector of length N,
which is the number of equations we have (which correspond to the number of pixels), and holds the derrivates 
of I1 with respect to x, "x" is dx, which is the value we are looking for,
and "y" is a vector which holds the differnces between each pixel: I2(xi)-I1(xi):

                * dI1/dx(x_1) *                * I2(x_1)-I1(x_1) *
               *      .        *  . dx  =     *         .         *
               *      .        *              *         .         *
                * dI1/dx(x_N) *                * I2(x_N)-I1(x_N) *    

The solution to this system of equations is : x = (A_transpose * A)^-1 * A_transpose * y
'''
def multiscale_iteration(I1, I2, dx):

    original_I1 = I1
    original_I2 = I2
    epochs = 50

    for epoch in range(epochs):
        # the function interpolate_by_dx is working with positive dx values, and we will send I1 or I2 accordingly.
        # we will decide which signal to move towards the other according to the sign of dx:
        if dx >= 0:
            I2 = interpolate_by_dx(I=original_I2, dx=dx)  # moving I2 "towards" I1
            I1 = original_I1[int(np.ceil(dx)):len(original_I1)]  # cropping I1 to be the same size as I2
        else:
            I1 = interpolate_by_dx(I=original_I1, dx=-dx)  # moving I2 "towards" I1
            I2 = original_I2[int(np.ceil(-dx)):len(original_I2)]  # cropping I1 to be the same size as I2

        # right hand side of the equation:
        y = I2 - I1
        # left hand side of the equation:
        A = np.convolve(I1, [1, 0, -1], mode='same')  # convolving is in reverse, so [1,0,-1] instead of [-1,0,1]
        # calculating the value of dx:
        dx = dx + (1 / (A.dot(A.transpose()))) * A.dot(y.transpose())

    return dx


############################ Program Flow ############################
if __name__ == "__main__":
    LOAD_FROM_CMD = True  # "True" if signals are received from cmd

    # load 2 signals. I2 is the shifted version of I1:
    if LOAD_FROM_CMD:
        I1 = np.load(sys.argv[1])['x1']
        I2 = np.load(sys.argv[2])['x2']

    else:
        I1 = np.load("data1d_1.npz")['x1']
        I2 = np.load("data1d_2.npz")['x2']

    # taking care of different lengths: #
    # crop I1 so it would be the same length as I2 (if I2 is shorter due to shifting process)
    # or crop I2 to be the same length of I1, depending on which signal is the shifted version of the other:

    if len(I1) > len(I2):
        I1 = I1[0:len(I2)]
    else:
        I2 = I2[0:len(I1)]

    '''
    SMOOTHING_FACTOR - the smoothing factor should increase for larger shifts between signals and reduced for smaller
    shifts between signals. There is a tradeoff between the accuracy of signal matching and the range of convergence - 
    if the smoothing factor is smaller, than the smoothed signals will be more similar to the original signals, and 
    therfore better accuracy in the matching process can be acheived. On the other hand, for larger shifts, an 
    approximation using only the first derivative is not good enough, unless we use proper (larger) smoothing 
    beforehand, which will give us a better range of convergence, in the expense of better acuuracy. 
    The default is our case is to use a smaller value of 0.5 for the SMOOTHING_FACTOR, since we got good results from
    smaller shifts (up to a few pixels). If the shift is larger, a bigger SMOOTHING_FACTOR should be considered.
    '''
    SMOOTHING_FACTOR = 0.5
    # first, let's smoothen the signals using a gaussian filter:
    I1 = gaussian_filter1d(I1, sigma=SMOOTHING_FACTOR)
    I2 = gaussian_filter1d(I2, sigma=SMOOTHING_FACTOR)

    # Different interpolation methods got the same results as simple subsampling, therefore we will use subsampling.
    # When using decimation, an additional filtering should be applied before each decimation:
    I1_decimated_half = gaussian_filter1d(I1, sigma=SMOOTHING_FACTOR)[0::2]
    I2_decimated_half = gaussian_filter1d(I2, sigma=SMOOTHING_FACTOR)[0::2]
    I1_decimated_quarter = gaussian_filter1d(I1_decimated_half, sigma=SMOOTHING_FACTOR)[0::2]
    I2_decimated_quarter = gaussian_filter1d(I2_decimated_half, sigma=SMOOTHING_FACTOR)[0::2]

    # first, running the iterative step on the quarter decimated signals:
    dx_quarter = multiscale_iteration(I1=I1_decimated_quarter, I2=I2_decimated_quarter, dx=0)
    # taking the result dx, multiplying it by 2, and using it for the iterative step on the half decimated signals:
    dx_half = multiscale_iteration(I1=I1_decimated_half, I2=I2_decimated_half, dx=2*dx_quarter)
    # taking the result dx, multiplying it by 2, and using it for the iterative step on the original signals:
    dx_final = multiscale_iteration(I1=I1, I2=I2, dx=2*dx_half)

    # printing the results:
    print(round(dx_final, 2))
