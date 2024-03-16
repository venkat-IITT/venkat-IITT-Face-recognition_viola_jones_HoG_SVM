import cv2 
import numpy as np
import matplotlib as plt 

img = cv2.imread("lenna.png", cv2.IMREAD_COLOR)

b1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
eroded1 = cv2.erode(b1, kernel, iterations=1)
eroded2 = cv2.erode(b1, kernel, iterations=2)

cv2.imshow("lenna",img)
cv2.imshow("lenna",eroded1)
cv2.waitKey(0)