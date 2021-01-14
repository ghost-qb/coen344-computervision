'''
 SANTA CLARA UNIVERSITY COMPUTER VISION I 
 PROGRAMMING ASSIGNMENT 1-1
 JANUARY 2021
 '''
 
# import the libraries we will need
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def drawbox(img, x0,y0, x1, y1, color):
    ''' DRAWBOX - a function to draw a colored rectangle into the image
    # img   - numpy tensor of the image
    # x0,y0 - upper left corner of box to draw
    # x1,y1 - lower right corner of box to draw
    # color - 'color' of line to draw. Should have same dimenspltn as pixel depth of img
    # return - none
    '''
    imgdim = img.shape
    xdim = imgdim[1] # note that x is usually defined as the horizontal/columns of the image. 
    ydim = imgdim[0] # note that y is usually defined as the vertical/rows of the image. And numpy images are stores as (rows, columns)

    # check to see if any parameters are out of range
    if x0<0 or x1>=xdim:
        print("drawbox: X  out of range")
        return
    if y0<0 or y1>=ydim:
        print("drawbox: Y  out of range")
        return
    if len(imgdim) > 2:
         if type(color) is not list or imgdim[2] != len(color):
            print("drawbox: 'color' is not of correct depth")
            return   
    # if color depth is greater than 1 then imgdim is greater than 2 and 'color' is a vector and not a scalar
    if len(imgdim) > 2:
        img[y0, x0:x1, :] = color
        img[y1, x0:x1, :] = color
        img[y0:y1, x0, :] = color
        img[y0:y1, x1, :] = color
        
    else: # scalar pixel depth
        img[y0, x0:x1] = color
        img[y1, x0:x1] = color
        img[y0:y1, x0] = color
        img[y0:y1, x1] = color
 
      
def simple_rotate(img, angle):
    ''' SIMPLE_ROTATE - simple function to rotate an image about its center
     img - input image to rotate
     angle - rotation angle, in radians
     returns an image the same size as the input (so may not fit)
    ''' 
    rotated_image = np.zeros_like(img) # same shape as input
    ydim, xdim = img.shape[:2]
    cx = xdim/2.0
    cy = ydim/2.0
    cosa = math.cos(angle)
    sina = math.sin(angle)
    for x in range(xdim):
        for y in range(ydim):
            fx = float(x)-cx
            fy = float(y)-cy
            ix = int( fx*cosa + fy*sina + cx)
            iy = int(-fx*sina + fy*cosa + cy)
            if ix>=0 and ix<xdim and iy>=0 and iy<ydim:
                rotated_image[y,x,:] = image[iy,ix,:]
                
    return rotated_image

''' start of main code
'''
        
# TODO: Choose your own color JPG image (of at least 600x400 pixels) and set 
# the path and filename here:
filename = "teslalogo.png"
dirname = "."
pathname = os.path.join(dirname, filename)

# load the image into a NUMPY array using matplotlib's imread functpltn
image = plt.imread(pathname)

if image is None:
    print("Unable to open ",pathname)
    exit(1)

# some systems load the image as a read-only array, set it to writeable
image.setflags(write=True)

# display the image
plt.imshow(image)
plt.title("Input Image")
plt.show()

# draw a red box onto the image (destructive)
drawbox(image, 10,10,100,100, [255,0,0])

# display it
plt.imshow(image)
plt.title("Image with Red Box")
plt.show()

# TODO: draw a CYAN rectangle from [20,50] to [80,80] and display it
# your code here
plt.imshow(image)
plt.plot([20,20, 80, 80, 20], [50,80, 80, 50, 50],color='c')
plt.title("Image with Cyan Box")
plt.show()

# create a new image by rotating the image about its center
image_rot = simple_rotate(image, math.pi/30)
# See http://scikit-image.org/docs/dev/api/skimage.transform.html?highlight=rotate#skimage.transform.rotate
# for a more sophisticed way to rotate an image

# Draw a box (non-destructive) on top of the rotated image
plt.imshow(image_rot)
plt.plot([20,20, 200, 200, 20], [30,100, 100, 30, 30]) # note that plot draws a series of lines between points
plt.title("Rotated Image with Drawn Box")
plt.show()

# TODO: save the rotated image in a new file (choose your own file name)
save_filename = "hw1p1-saved.jpg"
pathname = os.path.join(dirname, save_filename)
# convert to 8bit integer first
image_rot = (255*image_rot).astype('uint8')
plt.imsave(pathname, image_rot)

# TODO: Upload the saved image to Camino for credit

# create a unsigned 8-bit black and white version by summing R, G and B 
# as float values and casting the array back into uint8

float_image = image.astype(float)
image_bw = 255.0*(float_image[:,:,0] + float_image[:,:,1] + float_image[:,:,2])/3.0
image_bw = image_bw.astype('uint8')
                           
plt.imshow(image_bw,cmap=plt.cm.gray)
plt.title("Gray Scale Image")
plt.show()