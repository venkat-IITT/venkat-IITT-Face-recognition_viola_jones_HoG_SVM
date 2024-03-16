import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog
from skimage import exposure
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def viola_jones(grimg):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Detect faces in the image
    faces = face_cascade.detectMultiScale(grimg, scaleFactor=1.01, minNeighbors=6, minSize=(30, 30))
    cp_img = []  
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_roi = grimg[y:y+h, x:x+w]
        #cp_img.append(grimg[y:y+h, x:x+w])
        resized_face = cv2.resize(face_roi, (150, 150))
        hog_features, hog_image = hog(resized_face, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return resized_face, hog_features, hog_image

# testing
#img = plt.imread('lenna.png')

# Path to the folder containing images
folder_path = 'D:\\FER\\test\\DB\\Train'
# List to store images and corresponding labels (if available)

image_arrays = []
image_arrays_color = [];
crp_faces = []; 
HFS = [];
HIS = [];
Labels=[];
for filename in os.listdir(folder_path):
        # Load the image
        image = cv2.imread(os.path.join(folder_path, filename))
        lbl = filename[0:3]
        height, width, channels = image.shape
        grimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        Labels.append(lbl)  
        
        new_width = 100
        new_height = 150

        reshaped_image = cv2.resize(image, (new_width, new_height))
        image_arrays.append(reshaped_image)
        image_arrays_color.append(image)
        FD_image1,hf,hi = viola_jones(grimg)
        #FD_image2 = cv2.resize(FD_image1, (100,100))
        crp_faces.append(FD_image1)
        HFS.append(hf)
        HIS.append(hi)
        
# Display individual arrays size within the cell matrix
for i, image_array in enumerate(image_arrays):
    # Display the shape of each image
    print(f"Shape of image {i+1}:", image_array.shape)


#Testimage
Timage = cv2.imread('s10_05.jpg')
Tgrimage = cv2.cvtColor(Timage, cv2.COLOR_BGR2GRAY) 
T1image = cv2.resize(Timage,(new_width, new_height))
FD_image,hft,hit = viola_jones(Tgrimage)


#plt.imshow(hit)


X_train, X_test, y_train, y_test = train_test_split(HFS, Labels, test_size=0.3, random_state=42)

# Train SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)


# Predict labels for test data
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



'''
sub = [];
# Access individual arrays within the cell matrix        
print("Accessing individual arrays:")
for i, image_array in enumerate(HIS):
    img=crp_faces[i]
    sb = np.abs(hit - img)
    sbb = sum(sum(sb))
    sub.append(sbb)


# find the minimum 
mn = min(sub)
pos = np.argmin(sub)

rc_img = image_arrays_color[pos]

fig = plt.figure(figsize=(6,7)) 
r = 2
c = 2
fig.add_subplot(r,c, 1)
plt.imshow(Timage)
plt.title('Query Image')

fig.add_subplot(r,c,2)
plt.imshow(rc_img)
plt.title('Recognized Image')
'''

