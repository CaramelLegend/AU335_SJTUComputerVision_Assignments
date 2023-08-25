import math
import os
from xml.etree.ElementTree import ProcessingInstruction

import cv2
import numpy as np
from PIL import Image

path = "C:\\Users\\KenYuen\\OneDrive - sjtu.edu.cn\\Desktop\\UNI\\SEM 6 Spring Sem '22\\Computer Vision\\Assignment\\Assignment3\\Q5"
img = cv2.imread(os.path.join(path, 'Ques.jpg'))
img = cv2.resize(img,(640,480)) # 图像大小调整
img_copy = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度处理
ret, img_thres = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY) # 二值化
img_edges = cv2.Canny(img_thres, 100, 400, apertureSize = 3) # 边缘检测

# 霍夫变换与直线拟合
lines = cv2.HoughLines(img_edges, 1, np.pi/2, 20) 
starting_point = [] #直线起始点
ending_point = [] #直线总结点

#画线
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
        starting_point.append(pt1)
        ending_point.append(pt2)
        if(pt1[0] != pt2[0]): # 限制竖向直线的绘制
            cv2.line(img, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
cv2.imshow("houghline",img)
cv2.imwrite(os.path.join(path, 'img_Hough.jpg'), img)
# 霍夫变换与直线拟合完成


# 检测黑色起始像素点
black_pixels_front = []
for row in range(0, img_thres.shape[0]): # height
    for col in range(0, img_thres.shape[1]): # width
        if img_thres[row, col] < 127:
            black_pixels_front.append([row, col])
            break
# 黑色起始像素点检测完成

# 检测黑色像素点终结像素点
black_pixels_back = []
for row in range(0, img_thres.shape[0]): # height
    for col in range(0, img_thres.shape[1]): # width
        if img_thres[row, img_thres.shape[1]-col-1] < 127:
            black_pixels_back.append([row, img_thres.shape[1]-col-1])
            break
# 黑色终结像素点检测完成

"""
# 行间距为阈值进行像素聚类
points_in_a_line = 0
n = 0 # number of lines
average_starting_pixels = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]] # 每行文字中黑色像素平均起始点
col_start = []

for i in range(0, len(black_pixels_front)-1):
    if (i == len(black_pixels_front)-2): #最后一行
        #col_start.append(black_pixels_front[i][1])
        #min_col = min(col_start)
        points_in_a_line = points_in_a_line + 1
        average_starting_pixels[n][0] = (average_starting_pixels[n][0] + black_pixels_front[i][0])//points_in_a_line # 行起始
        #average_starting_pixels[n][1] = min_col
        average_starting_pixels[n][1] = (average_starting_pixels[n][1] + black_pixels_front[i][1])//points_in_a_line # 列起始
        points_in_a_line = 0
        break
    if black_pixels_front[i+1][0] - black_pixels_front[i][0] < 5:
        average_starting_pixels[n][0] = average_starting_pixels[n][0] + black_pixels_front[i][0]
        #col_start.append(black_pixels_front[i][1])
        average_starting_pixels[n][1] = average_starting_pixels[n][1] + black_pixels_front[i][1] 
        points_in_a_line = points_in_a_line + 1
    else:
        #col_start.append(black_pixels_front[i][1])
        #min_col = min(col_start)
        points_in_a_line = points_in_a_line + 1
        average_starting_pixels[n][0] = (average_starting_pixels[n][0] + black_pixels_front[i][0])//points_in_a_line # 行起始
        #average_starting_pixels[n][1] = min_col
        average_starting_pixels[n][1] = (average_starting_pixels[n][1] + black_pixels_front[i][1])//points_in_a_line # 列起始
        points_in_a_line = 0
        n = n + 1

points_in_a_line = 0
n = 0 # number of lines
average_ending_pixels = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]] # 每行文字中黑色像素平均终结点
col_end = []
for i in range(0, len(black_pixels_back)-1):
    if(i == len(black_pixels_back)-2): #最后一行
        #col_end.append(black_pixels_back[i][1])
        #max_col = max(col_end)
        points_in_a_line = points_in_a_line + 1
        average_ending_pixels[n][0] = (average_ending_pixels[n][0] + black_pixels_back[i][0])//points_in_a_line # 行起始
        #average_ending_pixels[n][1] = max_col
        average_ending_pixels[n][1] = (average_ending_pixels[n][1] + black_pixels_back[i][1])//points_in_a_line # 列起始
        points_in_a_line = 0
        break
    if black_pixels_back[i+1][0] - black_pixels_back[i][0] < 5:
        average_ending_pixels[n][0] = average_ending_pixels[n][0] + black_pixels_back[i][0] 
        #col_end.append(black_pixels_back[i][1])
        average_ending_pixels[n][1] = average_ending_pixels[n][1] + black_pixels_back[i][1] 
        points_in_a_line = points_in_a_line + 1
    else:
        #col_end.append(black_pixels_back[i][1])
        #max_col = max(col_end)
        points_in_a_line = points_in_a_line + 1
        average_ending_pixels[n][0] = (average_ending_pixels[n][0] + black_pixels_back[i][0])//points_in_a_line # 行终结
        #average_ending_pixels[n][1] = max_col
        average_ending_pixels[n][1] = (average_ending_pixels[n][1] + black_pixels_back[i][1])//points_in_a_line # 列终结
        points_in_a_line = 0
        n = n + 1
# 像素点聚类完成
"""

# 直线首尾点判断
adjusted_line_start = []
adjusted_line_end = []
for i in range (0, len(starting_point)):
    for j in range (0, len(black_pixels_front)):
        if starting_point[i][1] == black_pixels_front[j][0]: # 直线与起始黑色像素的行比较
            for k in range(0, len(black_pixels_back)):
                if starting_point[i][1] == black_pixels_back[k][0]: # 直线与终结黑色像素的行比较
                    adjusted_line_start.append(black_pixels_front[j])
                    adjusted_line_end.append(black_pixels_back[k])
# 直线首尾点判断完成

# 所有首尾点排序
for i in range(0, len(adjusted_line_start) - 1):
    for j in range (0, len(adjusted_line_start) - i - 1):
        if adjusted_line_start[j][0] > adjusted_line_start[j+1][0]:
            adjusted_line_start[j] , adjusted_line_start[j+1] = adjusted_line_start[j+1] , adjusted_line_start[j]

for i in range(0, len(adjusted_line_end) - 1):
    for j in range (0, len(adjusted_line_end) - i - 1):
        if adjusted_line_end[j][0] > adjusted_line_end[j+1][0]:
            adjusted_line_end[j] , adjusted_line_end[j+1] = adjusted_line_end[j+1] , adjusted_line_end[j]
# 排序完成

# 行间距为阈值进行直线聚类
lines_in_text = 0
n = 0 # number of lines
col_start = []
average_line_starting_pixels = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]] # 每行文字中黑色像素平均起始点
for i in range(0, len(adjusted_line_start)-1):
    if(i == len(adjusted_line_start)-2): #最后一行
        col_start.append(adjusted_line_start[i][1])
        print(col_start)
        min_col = min(col_start)
        lines_in_text = lines_in_text + 1
        average_line_starting_pixels[n][0] = (average_line_starting_pixels[n][0] + adjusted_line_start[i][0])//lines_in_text # 行起始
        average_line_starting_pixels[n][1] = min_col # 列起始
        lines_in_text = 0
        col_start.clear()
        break
    if adjusted_line_start[i+1][0] - adjusted_line_start[i][0] < 14:
        average_line_starting_pixels[n][0] = average_line_starting_pixels[n][0] + adjusted_line_start[i][0]
        col_start.append(adjusted_line_start[i][1])
        lines_in_text = lines_in_text + 1
    else:
        col_start.append(adjusted_line_start[i][1])
        min_col = min(col_start)
        lines_in_text = lines_in_text + 1
        average_line_starting_pixels[n][0] = (average_line_starting_pixels[n][0] + adjusted_line_start[i][0])//lines_in_text # 行起始
        average_line_starting_pixels[n][1] = min_col # 列起始 
        lines_in_text = 0
        n = n + 1
        col_start.clear()

lines_in_text = 0
n = 0 # number of lines
average_line_ending_pixels = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]] # 每行文字中黑色像素平均终结点
col_end = []
for i in range(0, len(adjusted_line_end)-1):
    if(i == len(adjusted_line_end)-2): #最后一行
        col_end.append(adjusted_line_end[i][1])
        max_col = max(col_end)
        lines_in_text = lines_in_text + 1
        average_line_ending_pixels[n][0] = (average_line_ending_pixels[n][0] + adjusted_line_end[i][0])/lines_in_text # 行终点
        average_line_ending_pixels[n][1] = max_col # 列终点
        lines_in_text = 0
        col_end.clear()
        break
    if adjusted_line_end[i+1][0] - adjusted_line_end[i][0] < 14:
        average_line_ending_pixels[n][0] = average_line_ending_pixels[n][0] + adjusted_line_end[i][0]
        col_end.append(adjusted_line_end[i][1])
        lines_in_text = lines_in_text + 1
    else:
        col_end.append(adjusted_line_end[i][1])
        max_col = max(col_end)
        lines_in_text = lines_in_text + 1
        average_line_ending_pixels[n][0] = (average_line_ending_pixels[n][0] + adjusted_line_end[i][0])//lines_in_text #行终点
        average_line_ending_pixels[n][1] = max_col # 列终点
        lines_in_text = 0
        n = n + 1
        col_end.clear()
# 直线聚类完成

final_line_start = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
final_line_end = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
for i in range(0,11):
    """
    final_line_start[i][1] = int((average_line_starting_pixels[i][0] + average_starting_pixels[i][0])//2) # 起始行
    final_line_start[i][0] = int((average_line_starting_pixels[i][1] + average_starting_pixels[i][1])//2) # 起始列
    final_line_end[i][1] = int((average_line_ending_pixels[i][0] + average_ending_pixels[i][0])//2) # 终点行
    final_line_end[i][0] = int((average_line_ending_pixels[i][1] + average_ending_pixels[i][1])//2) # 终点列
    """
    final_line_start[i][1] = int(average_line_starting_pixels[i][0]) # 起始行
    final_line_start[i][0] = int(average_line_starting_pixels[i][1]) # 起始列
    final_line_end[i][1] = int(average_line_ending_pixels[i][0]) # 终点行
    final_line_end[i][0] = int(average_line_ending_pixels[i][1]) # 终点列

for i in range (0, len(final_line_start)):
    cv2.line(img_copy, final_line_start[i], final_line_end[i], (0,0,0), 3, cv2.LINE_AA)
cv2.imshow('Adjusted Image', img_copy)
cv2.imwrite(os.path.join(path, 'img_Final.jpg'), img_copy)
cv2.waitKey()
cv2.destroyAllWindows()
