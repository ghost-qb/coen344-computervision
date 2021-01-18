#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
 SANTA CLARA UNIVERSITY COMPUTER VISION I 
 PROGRAMMING ASSIGNMENT 1-3
 JANUARY 2021
 '''
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.linalg import inv

def H_from_points(fp,tp):
  """ H_from_points
    Finds homography H, such that fp is mapped to tp
    using the linear DLT method. Points are conditioned
    automatically. 
    fp - the "from" points as a (2xn) numpy array
    to - the "to" points
    returns the 3x3 homography matrix 
    
    FROM: Programming Computer Vision with Python
    by Jan Erik Solem
    """
  if fp.shape != tp.shape:
    raise RuntimeError('number of points do not match')

  # condition points (important for numerical reasons)
  # --from points--
  m = np.mean(fp[:2], axis=1)
  maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
  C1 = np.diag([1/maxstd, 1/maxstd, 1])
  C1[0][2] = -m[0]/maxstd
  C1[1][2] = -m[1]/maxstd
  fp = np.dot(C1,fp)

  # --to points--
  m = np.mean(tp[:2], axis=1)
  maxstd = max(np.std(tp[:2], axis=1)) + 1e-9
  C2 = np.diag([1/maxstd, 1/maxstd, 1])
  C2[0][2] = -m[0]/maxstd
  C2[1][2] = -m[1]/maxstd
  tp = np.dot(C2,tp)

  # create matrix for linear method, 2 rows for each correspondence pair
  nbr_correspondences = fp.shape[1]
  A = np.zeros((2*nbr_correspondences,9))
  for i in range(nbr_correspondences):
    A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,
          tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
    A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,
          tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]

  U,S,V = np.linalg.svd(A)
  H = V[8].reshape((3,3))

  # decondition
  H = np.dot(np.linalg.inv(C2),np.dot(H,C1))

  # normalize and return
  return H / H[2,2]

def project_image_points(points):
  ''' project_image_points
  # function to convert a list of homogenous image coordinates to 
  # 3D camera coordintes for plotting
  '''
  xs = points[0,:]/points[2,:]
  ys = points[1,:]/points[2,:]
  return xs, ys

def putimage1intoimage2usingHomography(image1, image2, H):
    ''' putimage1intoimage2usingHomograpy
    A function to use the Homography matrix H to warp image 1 into image 2
    each pixel of image1 is warped into image2 coordinate space and if within
    range, copied to the closest integer position
    '''
    sy1, sx1 = image1.shape[:2]
    sy2, sx2 = image2.shape[:2]
    
    # loop over every pixel in image 1
    for x in range(sx1):
        for y in range(sy1):
            # create a homogeneous coordinate vector fom the pixel point
            p1 = np.array([x,y,1]).T
            # use the homography to compute the corresponding point in image2
            p2 = H.dot(p1)
            # convert to pixel and clip to nearest integer value
            x2,y2 = p2[:2]/p2[2]
            x2 = int(x2)
            y2 = int(y2)
            # if the point is inside image2, write over it
            if x2 > -1 and x2 < sx2:
                if y2 > -1 and y2 < sy2:
                    image2[y2,x2,:] = image1[y,x,:]
                    
def putimage1intoimage2usingHomographyINV(image1, image2, H):
    ''' putimage1intoimage2usingHomograpy
    A function to use the Homography matrix H to warp image 1 into image 2
    each pixel of image1 is warped into image2 coordinate space and if within
    range, copied to the closest integer position
    '''
    sy1, sx1 = image1.shape[:2]
    sy2, sx2 = image2.shape[:2]
    
    # loop over every pixel in image 1
    for x in range(sx2):
        for y in range(sy2):
            # create a homogeneous coordinate vector fom the pixel point
            p1 = np.array([x,y,1]).T
            # use the homography to compute the corresponding point in image2
            p2 = H.dot(p1)
            # convert to pixel and clip to nearest integer value
            x2,y2 = p2[:2]
            x2 = int(x2)
            y2 = int(y2)
            # if the point is inside image2, write over it
            if x2 > -1 and x2 < sx2:
                if y2 > -1 and y2 < sy2:
                    image1[y,x,:] = image2[y2,x2,:]
    
''' main code here
'''

# TODO: Set the path and filename for the sheetofpaper.jpg image you downloaded
# from the course site here:
filename = "sheetofpaper.jpg"
dirname = "./"
pathname = os.path.join(dirname, filename)

# load the image into a NUMPY array using matplotlib's imread function
image = plt.imread(pathname)

# plot it
plt.imshow(image)

# A sheet or paper is 8.5 by 11in. We can map inches to pixels in the homography
# The "from" points are the four corners of the paper in its coordinate system
# Set the points in any order you want, as long as you are consisten in the 
# ordering for the pixel points. For example if the lower left corner
# of the origin in paper space then fp[:,0] = [0,0,1] and tp[:,0] =[222,74,1]

# The edges of the paper in inches
fromp = np.array([[11,8.5,1],
              [11,0,1],
              [0,0,1],
              [0,8.5,1]])

# TODO: create the TO array for the corners in homogeneous pixel coordinates
top = np.array([[222,426,1],
                [752,222,1],
                [557,35,1],
                [86,183,1]])
    
# take the transpose as we want each point to be a Column, not a Row
fromp = fromp.T
top = top.T

# you can draw lines to the TO points on top of the paper to make sure you 
# got them right. they should connect the four corner points on the paper
xs, ys = project_image_points(top)
plt.plot(xs, ys)
              
# Now compute the homography
H = H_from_points(fromp,top)

# compare the computed TO points to the real TO points. They should be the same
xs, ys = project_image_points(top)
print("Inputed Points (X,Y):",xs,ys)
tp_compare = H.dot(fromp)
xs, ys = project_image_points(tp_compare)
print("Computed Points (X,Y):",xs,ys)

# now draw a grid of lines across the paper  
# across the short axis
fp = np.array([[2.75,0,1],
               [2.75,8.5,1],
               [5.5,8.5,1],
              [5.5,0,1],
              [8.25,0,1],
              [8.25,8.5,1]])

fp = fp.T
tp = H.dot(fp)
xs, ys = project_image_points(tp)
plt.plot(xs, ys)

# across the long axis
fp = np.array([[0,2,1],
              [11,2,1],
              [11,4,1],
              [0,4,1],
              [0,6,1],
              [11,6,1]])
fp = fp.T
tp = H.dot(fp)
xs, ys = project_image_points(tp)
plt.plot(xs, ys)
plt.imshow(image)
plt.show()

# TODO: enter the name and path for the bookcover image you downloaded
filename = "bookcover.jpg"
dirname = "./"
pathname = os.path.join(dirname, filename)
# load the image into a NUMPY array using matplotlib's imread function
book = plt.imread(pathname)

# plot it
plt.imshow(book)

# create FROM points which are now the four corners of the book cover
# which are the corners of the full images
# we've been using the "X" dimension as the horizontal or column and
# the "Y" dimension as the vertial or row. Note that np.shape has (Y,X) order
sy, sx = book.shape[:2]

fp = np.array([[0,0,1],
              [0,sy,1],
              [sx,sy,1],
              [sx,0,1]])

# take the transpose as we want each point to be a Column, not a Row
fp = fp.T

# the TO points are the same and are the edges of the paper

#compute the homography
H = H_from_points(fp,top)

H_inv = inv(H)

# Now use the Homography to map the book cover to the original image
image = np.array(image) # make a copy so we can write over it
book = np.array(book)
putimage1intoimage2usingHomographyINV(book, image, H_inv)
plt.imshow(image)
plt.show()

