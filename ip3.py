import cv2 
import numpy as np
import matplotlib as plt 

video = cv2.VideoCapture('sample.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
print('frames per second =',fps)
min =0
sec = 20
frame_id = int(fps*(min*60 + sec))
print('frame id =',frame_id)


