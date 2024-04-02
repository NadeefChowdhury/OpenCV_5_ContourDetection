import cv2 as cv
img = cv.imread('C:/Users/User 2/Desktop/Tyson.jpg')


#Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
cv.waitKey(0)

canny = cv.Canny(img, 125, 175)
cv.imshow('Canny', canny)
cv.waitKey(0)


contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print("Number of contours: " + str(len(contours)))


##BLURRING REDUCES THE NUMBER OF CONTOURS


##Threshing
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)
cv.waitKey(0)
contours2, hierarchies2 = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print("Number of contours: " + str(len(contours2)))


##Drawing Contours
import numpy as np
blank = np.zeros(img.shape, dtype='uint8')
cv.drawContours(blank, contours, -1, (255,255,255), 1)
cv.imshow('Contours', blank)
cv.waitKey(0)