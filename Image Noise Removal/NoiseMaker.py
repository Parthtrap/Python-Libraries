# -------------------- Making Salt and Pepper Noise --------------------#

# Import Library
import cv2
import numpy as np
import random

# Defining Global Variables
ImagePath = "./test.png"
noiseFactor = 0.1

# Defining Salting and Pepper Noise making function


def SaltAndPepperSprinkler(img, noiseFactor):
    # Code Here
    return img


# Read the image
img = cv2.imread(ImagePath, cv2.IMREAD_REDUCED_COLOR_2)
img2 = cv2.imread(ImagePath, cv2.IMREAD_REDUCED_GRAYSCALE_2)

# Convert to Noisy Image
# img3 = SaltAndPepperSprinkler(img, noiseFactor)
# img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

cv2.imshow("OG Image : ", img)
cv2.imshow("GrayScale : ", img2)
# cv2.imshow("Noisy Image : ", img3)
# cv2.imshow("Grayscale Noisy Image : ", img4)

cv2.waitKey(0)
cv2.destroyAllWindows()
