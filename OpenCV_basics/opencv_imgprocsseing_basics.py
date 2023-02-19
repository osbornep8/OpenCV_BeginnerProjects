import cv2
import numpy as np

#from google.colab.patches import cv2_imshow

img = cv2.imread('trial.jpg')
imgR = cv2.resize(img, (600,780))

# 1. Convert Image to Grayscale:
gray_img1 = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

#Total Image Size
print(img.size)

#You can access a pixel value by its row and column coordinates. For BGR image, it returns an array of Blue, Green, Red values. For grayscale image, just corresponding intensity is returned.
px = img[100,100]
gray_px = gray_img1[100,100]
print(px)
print(gray_px)



# imshow() is used to display the ouput in a new window
#The imshow() function is designed to be used along  with the waitKey() and destroyAllWindows() / destroyWindow() functions. To display multiple images at once, specify a new window name for every image you want to display. 
cv2.imshow('Frame',gray_img1)
cv2.waitKey(0)
cv2.destroyWindow('Frame')

#imwrite() function is used to save/write the ouput in the file directory
cv2.imwrite('grayscale.jpg', gray_img1)


# 2. Edge Deetection:


# Preprocessing - Blur the image for better edge detection (This is done to reduce the noise in the image)
img_blur = cv2.GaussianBlur(gray_img1, (5,5), 0)

# (i) Using Sobel (detects edges marked by sudden changes in pixel intensity)

edge_sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
#edge_sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
#edge_sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
cv2.imshow("Sobel Edge Detection", edge_sobelx)
cv2.waitKey(0)
cv2.destroyWindow('Sobel Edge Detection')

# (ii) Using Canny - includes not only Sobel Edge Detection but also Non-Maximum Suppression and Hysteresis Thresholding.
edge_det = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
cv2.imshow('Canny Edge Detection', edge_det)
cv2.waitKey(0)
cv2.destroyWindow('Canny Edge Detection')
cv2.imwrite('Canny_Edge_det.jpg', edge_det)


# 3. Erosion:

kernels = np.ones((5,5), np.uint8)

img_erosion = cv2.erode(imgR, kernel=kernels, iterations=1)

cv2.imshow("Erosion", img_erosion)
cv2.waitKey(0)
cv2.destroyWindow('Erosion')


# 4. Dilation:

img_dilate = cv2.dilate(imgR, kernels, iterations=1)

cv2.imshow("Dilation", img_dilate)
cv2.waitKey(0)
cv2.destroyWindow('Dilation')

