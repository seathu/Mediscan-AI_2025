# code to call python from flask
#  from subprocess import call
# call(["python", "crop.py"])

# list of all files in a folder
#  import os
# path = "mongo"
# dir_list = os.listdir(path)
# print("Files and directories in '", path, "' :")
# print(dir_list)

# clear the contents of folder
# import os
# folder_path = r"mongo"
# for filename in os.listdir(folder_path):
#    file_path = os.path.join(folder_path, filename)
#    if os.path.isfile(file_path):
#       os.remove(file_path)      

# image enhancement

import cv2
import matplotlib.pyplot as plt
import numpy as np
def bgremove3(myimage):    
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)     
    s = myimage_hsv[:,:,1]
    s = np.where(s < 127, 0, 1) 
    v = (myimage_hsv[:,:,2] + 127) % 255
    v = np.where(v > 127, 1, 0)      
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  
    background = np.where(foreground==0,255,0).astype(np.uint8) 
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    foreground=cv2.bitwise_and(myimage,myimage,mask=foreground)
    finalimage = background+foreground 
    return finalimage


# image = cv2.imread("E:\\karthik_drugs_recomendations\\mongo\\images\\prescriptions\\1.jpg")
image = cv2.imread("E:\\karthik_drugs_recomendations\\mongo\\mongo\\new0.jpg")
g_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('enhance/original.jpg', image)
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale Image', g_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# image = cv2.imread("E:\\karthik_drugs_recomendations\\mongo\\images\\prescriptions\\2.jpg")
# image = cv2.imread('enhance/original.jpg')
#############
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(3, 4, 1)
plt.title("Original")
plt.imshow(image)
brightness = 10 
contrast = 2.3  
image2 = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)
plt.subplot(3, 4, 2)
plt.title("Brightness & contrast1")
plt.imshow(image2)
alpha = 1.5 
beta = 50  
image2 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
plt.subplot(3, 4, 3)
plt.title("Brightness & contrast2")
plt.imshow(image2)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, kernel)
plt.subplot(3, 4, 4)
plt.title("Sharpening")
plt.imshow(sharpened_image)
plt.subplot(3, 4, 5)
sharpened_image2 = cv2.Laplacian(image, cv2.CV_64F)
plt.title("Laplacian Sharpening")
plt.imshow(sharpened_image2)
filtered_image = cv2.medianBlur(image, 11)
plt.subplot(3, 4, 6)
plt.title("Median Blur")
plt.imshow(filtered_image)
filtered_image2 = cv2.GaussianBlur(image, (7, 7), 0)
plt.subplot(3, 4, 7)
plt.title("Gaussian Blur")
plt.imshow(filtered_image2)
image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
plt.subplot(3, 4, 8)
plt.title("enhanced coloured")
plt.imshow(image2)
resized_image = cv2.resize(image, (2100, 1500))
plt.subplot(3, 4, 9)
plt.title("Resized")
plt.imshow(resized_image)
scaled_image = cv2.resize(image, None, fx=2, fy=2)
cv2.imwrite('Scaled.jpg', scaled_image)
plt.subplot(3, 4, 10)
plt.title("Scaled")
plt.imshow(scaled_image)
inverse_image = 255 - image
plt.subplot(3, 4, 11)
plt.title("Inverse color")
plt.imshow(inverse_image)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized_image = cv2.equalizeHist(gray_image)
cv2.imwrite('equalized.jpg', equalized_image)
plt.subplot(3, 4, 12)
plt.title("equalized")
plt.imshow(equalized_image)
plt.tight_layout()
plt.show()
