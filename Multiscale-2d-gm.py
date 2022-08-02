import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import sys
from scipy import ndimage


# function which interpolates the image I by dx, dy
def interpolate_by_dx_dy(I, dx, dy):  # dx - movement between cols, dy - movement between rows
    full_shifted = np.zeros(I.shape)
    int_shifted = ndimage.shift(I, [np.fix(dy), np.fix(dx)])  # first, shift by the nearest integer
    remainder_x = dx - np.fix(dx)  # between -1 and 1
    remainder_y = dy - np.fix(dy)  # between -1 and 1
    height, width = I.shape
    # after shifting by integer, we interpolate with the remaining dx and dy:
    for col in range(int(np.ceil(remainder_x)), width+int(np.floor(remainder_x))):
        for row in range(int(np.ceil(remainder_y)), height+int(np.floor(remainder_y))):
            full_shifted[row][col] = \
            int_shifted[row][col] * (1-np.abs(remainder_x))*(1-np.abs(remainder_y)) + \
            int_shifted[row-int(np.ceil(remainder_y)+np.floor(remainder_y))][col]*(1-np.abs(remainder_x))*(np.abs(remainder_y)) + \
            int_shifted[row][col-int(np.ceil(remainder_x)+np.floor(remainder_x))]*(np.abs(remainder_x))*(1-np.abs(remainder_y)) + \
            int_shifted[row-int(np.ceil(remainder_y)+np.floor(remainder_y))][col-int(np.ceil(remainder_x)+np.floor(remainder_x))]\
            *(np.abs(remainder_x))*(np.abs(remainder_y))

    # cropping the result matrix to only include relevant data:
    if dx >= 0:
        full_shifted = full_shifted[:, int(np.ceil(dx)):width]
    else:
        full_shifted = full_shifted[:, 0:width + int(np.floor(dx))]
    if dy >= 0:
        full_shifted = full_shifted[int(np.ceil(dy)):height, :]
    else:
        full_shifted = full_shifted[0:height + int(np.floor(dy)), :]

    # return the cropped interpolated image:
    return full_shifted


'''This following function performs the same functionality as the original "iterative-2d-gm", but with some initial
d_total (which can be different from [0, 0]).

We know that I2(xi, yi) = I1(xi+dx,yi+dy) for some dx,dy which we would like to estimate.
We would now want to create a system of linear equations which we will solve using LS (least squares).
The system of equations is of the following format: Ax = y, where "A" (in our case is) vector of length N,
which is the number of equations we have (which correspond to the number of pixels), and holds the derrivates 
of I1 with respect to x, and the derivatives of I1 with respect to y, "x" is (dx,dy) which is the value we are 
looking for, and "y" is a vector which holds the differnces between each pixel: I2(xi,yi)-I1(xi,yi):

                * dI1(x_1,y_1) dI1(x_1,y_1)*                * I2(x_1,y_1)-I1(x_1,y_1) *
               *     dx      .         dy   *  .* dx * =   *             .             *
               *             .              *   * dy *     *             .             *
               * dI1(x_N,y_N) dI1(x_N,y_N)  *               * I2(x_N,y_N)-I1(x_N,y_N) *    
                *    dx             dy     *

The solution to this system of equations is : x = (A_transpose * A)^-1 * A_transpose * y
'''
def multiscale_iteration(I1, I2, d_total):

    original_I1 = I1
    original_I2 = I2
    height, width = I1.shape
    epochs = 30

    for epoch in range(epochs):

        A1 = np.array([])
        A2 = np.array([])

        I2 = interpolate_by_dx_dy(I=original_I2, dx=d_total[0], dy=d_total[1])  # moving I2 "towards" I1

        # crop I1 to be the same shape as I2:
        if d_total[0] >= 0:
            I1 = original_I1[:, int(np.ceil(d_total[0])):width]
        else:
            I1 = original_I1[:, 0:width+int(np.floor(d_total[0]))]
        if d_total[1] >= 0:
            I1 = I1[int(np.ceil(d_total[1])):height, :]
        else:
            I1 = I1[0:height+int(np.floor(d_total[1])), :]

        # right hand side of the equation:
        y = I2.flatten() - I1.flatten()

        # calculate dI1/dx (which will be stored in A1) and calculate dI1/dy (which will be stored in A2_flat):
        new_height, new_width = I1.shape
        A2 = np.zeros((new_width, new_height))
        A2_flat = np.array([])
        for i in range(new_height):
            A1 = np.append(A1, np.squeeze(np.convolve(I1[i], [1, 0, -1], mode='same')))
        for j in range(new_width):
            A2[j] = np.convolve(np.transpose(I1)[j], [1, 0, -1], mode='same')
        A2 = np.transpose(A2)
        for i in range(new_height):
            A2_flat = np.append(A2_flat, A2[i])

        # left hand side of the equation:
        A = np.transpose([A1, A2_flat])

        # calculating the value of d_total = [dx, dy]:
        d_total = d_total + np.matmul(np.linalg.inv(np.matmul(A.transpose(), A)), np.matmul(np.transpose(A), y))
    return d_total


############################ Program Flow ############################
if __name__ == "__main__":

    LOAD_FROM_CMD = True  # "True" if images are received from cmd

    # load 2 signals. I2 is the shifted version of I1:
    if LOAD_FROM_CMD:
        I1 = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)
        I2 = cv2.cvtColor(cv2.imread(sys.argv[2]), cv2.COLOR_BGR2GRAY)
    else:
        I1 = cv2.cvtColor(cv2.imread('im1.bmp'), cv2.COLOR_BGR2GRAY)
        I2 = cv2.cvtColor(cv2.imread('im2.bmp'), cv2.COLOR_BGR2GRAY)

    # taking care of different shape case:
    height1, width1 = I1.shape
    height2, width2 = I2.shape

    if width1 > width2:
        I1 = I1[:, 0:width2]
    else:
        I2 = I2[:, 0:width1]
    if height1 > height2:
        I1 = I1[0:height2, :]
    else:
        I2 = I2[0:height1, :]

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
    I1 = gaussian_filter(I1, sigma=SMOOTHING_FACTOR)
    I2 = gaussian_filter(I2, sigma=SMOOTHING_FACTOR)

    # Different interpolation methods got the same results as simple subsampling, therefore we will use subsampling.
    # The different interpolation methods can be seen in the following comment:
    ######################################################################
    # width_quarter = int(I1.shape[1] * 0.25)
    # width_half = int(I1.shape[1] * 0.5)
    # height_quarter = int(I1.shape[0] * 0.25)
    # height_half = int(I1.shape[0] * 0.5)
    # dim_quarter = (width_quarter, height_quarter)
    # dim_half = (width_half, height_half)
    #
    # # resize options: INTER_NEAREST, INTER_LINEAR (default), INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
    # I1_decimated_quarter = cv2.resize(I1, dim_quarter, interpolation=cv2.INTER_CUBIC)
    # I1_decimated_half = cv2.resize(I1, dim_half, interpolation=cv2.INTER_CUBIC)
    # I2_decimated_quarter = cv2.resize(I2, dim_quarter, interpolation=cv2.INTER_CUBIC)
    # I2_decimated_half = cv2.resize(I2, dim_half, interpolation=cv2.INTER_CUBIC)
    ######################################################################

    # Using default subsampling and smoothening new signals:
    # When using decimation, an additional filtering should be applied before each decimation:
    I1_decimated_half = gaussian_filter(I1, sigma=SMOOTHING_FACTOR)[0::2, 0::2]
    I2_decimated_half = gaussian_filter(I2, sigma=SMOOTHING_FACTOR)[0::2, 0::2]
    I1_decimated_quarter = gaussian_filter(I1_decimated_half, sigma=SMOOTHING_FACTOR)[0::2, 0::2]
    I2_decimated_quarter = gaussian_filter(I2_decimated_half, sigma=SMOOTHING_FACTOR)[0::2, 0::2]

    # first, running the iterative step on the quarter decimated images:
    d_total_quarter = multiscale_iteration(I1=I1_decimated_quarter, I2=I2_decimated_quarter, d_total=[0,0])
    # taking the result d_total, multiplying it by 2, and using it for the iterative step on the half decimated images:
    d_total_half = multiscale_iteration(I1=I1_decimated_half, I2=I2_decimated_half, d_total=2*d_total_quarter)
    # taking the result d_total, multiplying it by 2, and using it for the iterative step on the original images:
    d_total_final = multiscale_iteration(I1=I1, I2=I2, d_total=2*d_total_half)

    # printing the results:
    print(str(round(d_total_final[0], 2)) + " " + str(round(d_total_final[1], 2)))

