A FEW NOTES:

1) The programs work as was requested in the assignment (using the command line).
2) Better results can be achieved when using longer signals/bigger images, since there are more equations with the same number of 
   variables (dx for the 1d case, or dx and dy	for the 2d case).
3) For the multiscale method, few different image resizing methods were tested (and can be seen in a comment in the
   "Multiscale-2d-gm.py" file), but all of them got quite similar results to simple sub-sampling, which we eventually used.
4) A default value of 50 epochs was chosen as the number of iterations we perform until we get to a "steady state". Usually 
   the algorithms converge beforehand. The number of epochs can be changed using the "epochs" variable. It should be noted 
   that for large signals/images (such as the"lena.bmp" we used for testing) the program might take a few minutes to run 
   (the Multiscale-2d-gm takes the longest time to run). To reduce runtime the number of epochs can be reduced.
5) SMOOTHING_FACTOR - the smoothing factor should increase for larger shifts between signals/images and reduced for smaller
   shifts between signals. There is a tradeoff between the accuracy of signal/image matching and the range of convergence - 
   if the smoothing factor is smaller, than the smoothed signals/images will be more similar to the original signals/images, and 
   therfore better accuracy in the matching process can be achieved. On the other hand, for larger shifts, an 
   approximation using only the first derivative is not good enough, unless we use proper (larger) smoothing 
   beforehand, which will give us a better range of convergence, in the expense of better acuuracy. 
   The default is our case is to use a smaller value of 0.5 for the SMOOTHING_FACTOR, since we got good results for
   smaller shifts (up to a few pixels). If the shift is larger (and this fact is known beforehand), a bigger SMOOTHING_FACTOR
   should be considered.
6) Size of signals/images - although it is given as a fact that the input signals/images are of the same size/shape, we also made  
   sure the algorithms work correctly even in the case of signals/images of different sizes/shapes.