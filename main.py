import cv2
import numpy as np
import imutils
import easyocr
import os
from matplotlib import pyplot as pl
image = cv2.imread('images/card/Ap.jpg')
img = cv2.imread('images/ace.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_filter = cv2.bilateralFilter(gray, 11, 15, 20)

edges= cv2.Canny(img_filter, 30, 200)
cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

coord = None
for i in cnts:
    prim = cv2.approxPolyDP(i, 5, True)
    if len(prim) == 4:
        coord = prim
        break
directory = 'images/card'


'''        
mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [coord], 0, 800, -1)
obr_img = cv2.bitwise_and(img, img, mask = mask)
'''
correct_matches_dct = {}
#for image in os.listdir(directory):
img2 = cv2.imread(directory+gray2, 0)
orb = cv2.ORB_create()
#gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kp1, des1 = orb.detectAndCompute(img, None)
kp2, des2 = orb.detectAndCompute(img2, None)
print(des2)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
correct_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        correct_matches.append([m])
        correct_matches_dct[image.split('.')[0]] = len(correct_matches)
        correct_matches_dct = dict(sorted(correct_matches_dct.items(),
        key=lambda item: item[1], reverse=True))


pl.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
pl.show()
