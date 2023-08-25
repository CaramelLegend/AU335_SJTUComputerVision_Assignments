import cv2
import numpy as np
import os
import math

path = "C:\\Users\\KenYuen\\OneDrive - sjtu.edu.cn\\Desktop\\UNI\\SEM 6 Spring Sem '22\\Computer Vision\\Assignment\\Assignment3\\Q4"
img = cv2.imread(os.path.join(path, 'solar.jpg'))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Image', img_gray)
cv2.imwrite(os.path.join(path, 'solarpanel.jpg'), img_gray)
img_edges = cv2.Canny(img_gray, 100, 400, apertureSize = 3)
# cv2.imshow('Canny Image', img_edges)
cv2.imwrite(os.path.join(path, 'solarpanel_Canny.jpg'), img_edges)

lines = cv2.HoughLines(img_edges, 1, np.pi/6, 100)
print(len(lines))

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
 
# cv2.imshow("houghline",img)
cv2.imwrite(os.path.join(path, 'solarpanel_Hough.jpg'), img)
# cv2.waitKey()
# cv2.destroyAllWindows()