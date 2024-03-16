import cv2 
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt 

img = cv2.imread("lenna.png", cv2.IMREAD_COLOR)
plt.figure()
plt.imshow(img)
b1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(b1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
eroded1 = cv2.erode(b1, kernel, iterations=1)
eroded2 = cv2.erode(b1, kernel, iterations=2)

fig = plt.figure(figsize=(6,7)) 
r = 2
c = 2

fig.add_subplot(r,c, 1)
plt.imshow(img)
plt.axis('off')
plt.title('First')

fig.add_subplot(r,c, 2)
plt.imshow(b1)
plt.axis('off')
plt.title('Second')

fig.add_subplot(r,c, 3)
plt.imshow(eroded1)
plt.axis('off')
plt.title('Third')

fig.add_subplot(r,c, 4)
plt.imshow(eroded2)
plt.axis('off')
plt.title('Fourth')

