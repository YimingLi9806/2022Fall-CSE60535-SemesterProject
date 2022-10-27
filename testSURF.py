# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2021

import cv2
import numpy as np
import math
from ROIPoly import roiPoly

# Read the input image
I = cv2.imread('nd2.jpg')

# Calculate SURF for your image
grayImage = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, nOctaves=3, nOctaveLayers=3) #cv2.xfeatures2d.SIFT_create()

kp, des = surf.detectAndCompute(grayImage, None)

# We can check how many SURF keypoints were found:
NoOfKeypoints, NoOfAttributes = len(kp), surf.descriptorSize()

print('I found ', NoOfKeypoints, ' SURF keypoints in this image.')

# Do you remember how many attributes each SURF keypoint descriptor should have? Let's check it:
print('Each SURF keypoints has ', NoOfAttributes, ' attributes.')

# Sort the points by response
def kp_response(kp):
    return kp.response
kp = sorted(kp, key = kp_response, reverse=True)

# Plot the N strongest points
N = 30
kp_filtered = []
for i in range(N):
    kp_filtered.append(kp[i])

# Auxiliary function to show the keypoints on the image
def drawKeypoints(img, kp_filtered, color=(0,255,0)):
    for kp in kp_filtered:
        center = (round(kp.pt[0]), round(kp.pt[1]))
        img = cv2.line(img, (center[0]-4, center[1]), (center[0]+4, center[1]), color=color, thickness=1)
        img = cv2.line(img, (center[0], center[1]-4), (center[0], center[1]+4), color=color, thickness=1)
        img = cv2.circle(img, center, radius=round(kp.size/2), color=color, thickness=1)
        theta = (math.pi * kp.angle)/180
        x1, y1 = center
        x2 = x1 + round(kp.size/2) * math.cos(theta)
        y2 = y1 + round(kp.size/2) * math.sin(theta)
        end_point = (round(x2), round(y2))
        img = cv2.line(img, center, end_point, color=color, thickness=1)
    return img

dark_grayImage = (grayImage*0.5).astype(np.uint8)
grayImage_with_kp = drawKeypoints(cv2.cvtColor(dark_grayImage, cv2.COLOR_GRAY2BGR), kp_filtered, color=(0,255,0)) #cv2.drawKeypoints
cv2.imshow('Image with Keypoints', grayImage_with_kp)

cv2.waitKey(0)
cv2.destroyAllWindows()


