import cv2
import os
import numpy as np
# Path to the folder containing images
folder_path = 'D:\\FER\\test\\DB'
# List to store images and corresponding labels (if available)
images = []
labels = []  # Optional
# Loop through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image = cv2.imread(os.path.join(folder_path, filename))
        height, width, channels = image.shape
        # Print the size of the image
        print("Width:", width)  
        print("Height:", height)

        new_width = 100
        new_height = 200

        reshaped_image = cv2.resize(image, (new_width, new_height))
        # Optional: Extract label from filename and append to labels list
        #label = filename.split('_')[0]  # Example: assuming filenames are like "label_image.jpg"
       # print("Label:", label) 
        #labels.append(label)

        # Append the image to the images list
        images.append(image)
        # Convert list of images to numpy array
#images1 = np.array(reshaped_image)