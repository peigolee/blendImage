import numpy as np
import cv2 as cv
from blend_modes import blend_modes

mask = cv.imread("mask.png")
foreground = cv.imread("foreground.jpg")
background = cv.imread("background.jpeg")

mask = cv.resize(mask, (256,256))
foreground = cv.resize(foreground, (256,256))
background = cv.resize(background, (256,256))


ratio = mask.astype(float) / 255
f_img = cv.multiply( 1 - ratio, foreground.astype(float))
b_img = cv.multiply( ratio, background.astype(float))
no_alpha_img = cv.add(f_img, b_img)
cv.imshow("normal blend", no_alpha_img/255)

b, g, r = cv.split(mask)
a = np.ones(b.shape, dtype=b.dtype) * 255 #creating a dummy alpha channel image.
mask = cv.merge((b,g,r,a))


b, g, r = cv.split(foreground)
a, _, _, _ = cv.split(mask) #creating a dummy alpha channel image.
foreground = cv.merge((b,g,r, 255 - a))
cv.imshow("a",a)
cv.imshow("foreground", foreground)


b, g, r = cv.split(background)
a = np.ones(b.shape, dtype=b.dtype) * 255 #creating a dummy alpha channel image.
background = cv.merge((b,g,r,a))
cv.imshow("background", background)


opacity = 0.9  # The opacity of the foreground that is blended onto the background is 70 %.
blended_img_float = blend_modes.hard_light(background.astype(float), foreground.astype(float), opacity)
cv.imshow("blended_img_float", blended_img_float/255)
cv.waitKey(0)


"""
mask = mask.astype(float)/255


f_img = cv.multiply( mask, foreground)
b_img = cv.multiply( 1 - mask, background)
no_alpha_img = cv.add(f_img, b_img)

img = cv.add(f_img, b_img)


cv.imshow("no_alpga", no_alpha_img/255)
cv.imshow("img", img/255)
cv.waitKey(0)
"""
