'''
 SANTA CLARA UNIVERSITY COMPUTER VISION I 
 PROGRAMMING ASSIGNMENT 1-2
 JANUARY 2021
 '''
 
# import the libraries we will need
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# MAKE_3DROT_MAT - makes a 3D Rotational Matrix
# given a rotation axis and angle
def make_3Drot_mat(axis, angle):
    eyemat = np.eye(3)
    (nx, ny, nz) = axis
    
    ncross = np.array([[0, -nz, ny],
                       [nz, 0, -nx],
                       [-ny, nx, 0]])
    ncross2 = ncross.dot(ncross)
    return eyemat + np.sin(angle)*ncross + (1-np.cos(angle))*ncross2

# MAKE_4DRT_MAT - makes 4D Rotation and Translation Matrix
# given a rotation axis and angle and 3D translation vector
def make_RT_mat(axis, angle, trans):
    answer = np.zeros([3,4])
    
    # make a 3x3 rotation matrix
    threeDrot_mat = make_3Drot_mat(axis, angle)
    # set the  left 3 columns of the RT matrix to the 3x3 rotation matrix
    answer[:3, :3] = threeDrot_mat;
    # Set the 4th column to the translation vector
    answer[:3, 3] = trans
    return answer

# function to convert a list of homogenous image coordinates to 
# 3D camera coordintes for plotting
def project_image_points(points):
    xs = points[0,:]/points[2,:]
    ys = points[1,:]/points[2,:]
    return xs, ys


# MAKE_CUBE
# this creates a series of points on a cube in homogenous coordinates
def make_cube(w):
    points = np.array([[0,0,0,1],
                   [w,0,0,1],
                   [w,w,0,1],
                   [0,w,0,1],
                   [0,0,0,1],
                   [0,0,w,1],
                   [0,w,w,1],
                   [w,w,w,1],
                   [w,0,w,1],
                   [0,0,w,1],
                   [0,0,0,1],
                   [w,0,0,1],
                   [w,0,w,1],
                   [w,w,w,1],
                   [w,w,0,1],
                   [0,w,0,1],
                   [0,w,w,1]])

    #    we want the points to be column vectors, so take the transpose
    points = points.T
    return(points)


''' main code starts here
'''
# make a numpy array of points on a unit cube
points_3D = make_cube(1)

# create a camera intrinsics matrix K
fx = 1000
fy = 1000
cx = 320
cy = 240
    
K = np.array([[fx,  0, cx],
              [0 , fy, cy],
              [0 ,  0,  1]])

# create a camera extrinsics matrix: 3D rigid motion matrix with a rotation and translation
rotAxis = [0,0,1]
rotAngle = 0.0
trans =   [2,-2,12]
RT = make_RT_mat(rotAxis, rotAngle, trans)

# create a projection matrix by combining K and RT
P = K.dot(RT)

# Now project the points from world coordinates to camera homogenous coordinates
points_image = P.dot(points_3D)

# Project camera homogenous coordintes to 2d image coordintes
xs, ys = project_image_points(points_image)

# plot the image points
fig, ax = plt.subplots()
plt.plot(xs,ys)

# make another cube, this time rotated around the Z axis and 2m above the camera
RT = make_RT_mat([0,0,1], math.pi/4, [-1.5,2,14])

# and rotate the cube points by RT and project to camera homogenous points
points_image = K.dot(RT.dot(points_3D))

xs, ys = project_image_points(points_image)
plt.plot(xs,ys)

# TODO make another cube, this time rotated around the Y axis by pi/4 and translated by X=-2.5, Y=-2, Z=18
RT = make_RT_mat([0,1,0], math.pi/4, [-2.5,-2,18])

# and rotate the cube points by RT and project to camera homogenous points
points_image = K.dot(RT.dot(points_3D))


xs, ys = project_image_points(points_image)
plt.plot(xs,ys)

# Now make two lane markers on the ground (Y=-2)
W = 1.5 # half width of the lanes
R = 1000 # radius of curvature of the lanes

# parameterization of angle theta by 0.01 radians
t = np.arange(math.pi/5000, math.pi/8, 0.01) 

# We will make the points directly into camera coordinates (no need for homogenous coords)
# left lane
xr = R - (R+W)*np.cos(t)
yr = -2 
zr = (R+W)*np.sin(t) 

# project to image points
xs = fx*xr/zr + cx
ys = fy*yr/zr + cy

plt.plot(xs,ys)

# right lane
xl = R - (R-W)*np.cos(t)
yl = -2 
zl = (R-W)*np.sin(t) 

# project to image points
xs = fx*xl/zl + cx
ys = fy*yl/zl + cy

plt.plot(xs,ys)

# TODO make two more lane markers to the left and right
# note that W is half the lane width
# left lane
xl_added = R - (R+2.5*W)*np.cos(t)
yl_added = -2 
zl_added = (R-2.5*W)*np.sin(t) 
# project to image points
xs = fx*xl_added/zl_added + cx
ys = fy*yl_added/zl_added + cy

plt.plot(xs,ys)

# right lane
xr_added = R - (R-2.5*W)*np.cos(t)
yr_added = -2 
zr_added = (R+2.5*W)*np.sin(t) 
# project to image points
xs = fx*xr_added/zr_added + cx
ys = fy*yr_added/zr_added + cy

plt.plot(xs,ys)

# finally set the image size limits and show the plot
plt.xlim(0,640)
plt.ylim(0,480)
plt.show()

# TODO: save the plot to a new file (choose your own file name)
# the following code SHOULD work, but not on all Mac's or PC's. If it doesn't 
# work (i.e. the resulting file is blank), then click on the the floppy disk icon below the plot
# and save the figure manually
save_filename = "draw_cubes_and_lanes.png"
dirname = "./"
pathname = os.path.join(dirname, save_filename)
plt.gcf()
plt.savefig(pathname)

# TODO: Upload the saved image to Camino for credit