import cv2
import numpy as np
import time
import math
import os
from sys import exit
# 设置一下小黑人脸图片的路径
img_R = cv2.imread("E:/wechat_jump/t_r.jpg", 0)
w_R, h_R = img_R.shape[::-1]
# 计算小人位置
def findHeadPoint(imgOld):
    # 模板匹配，检测坐标
    res = cv2.matchTemplate(imgOld, img_R, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return min_loc[0] + (w_R / 2), min_loc[1] + (h_R / 2)
# 计算踏板位置
def findNextPoint(imgGray, imgSrcStrong, w_screen, max_cut_high, x_R):
    # 截取图像关键部分
    img = imgGray[470:int(max_cut_high), 0:w_screen]
    # 截取对比度增强图片
    imgsrccut = imgSrcStrong[470:int(max_cut_high), 0:w_screen]
    max_w, max_high = img.shape[::-1]
    # 计算背景阈值
    maxVal = img[0][0] + 5
    minVal = maxVal - 20
    # gamma校正
    lut = np.zeros(256, dtype=img.dtype)
    for i, v in enumerate(lut):
        if (i >= minVal) and (i <= maxVal):
            lut[i] = 0
        else:
            lut[i] = 255
    img = cv2.LUT(img, lut)
    # 轮廓检测
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 计算中心线
    x_mid = w_screen // 2
    # 最高点
    x_up = 0
    y_up = 1500
    for contour in contours:
        for pt in contour:
            x = pt[0][0]
            # 忽略边缘噪声
            if x > (w_screen - 10) or x < 10:
                continue
            # 板子坐标和猪脚坐标要在中心线两侧
            if (x_R > x_mid) and (x > x_mid):
                continue
            if (x_R < x_mid) and (x < x_mid):
                continue
            # 由于下一个板子太矮并且和主角太近，把猪脚身子当成下一个板子了，距离猪脚过近的坐标都不能要
            if (x_R > x_mid) and (x > (x_R - 45)):
                continue
            if (x_R < x_mid) and (x < (x_R + 45)):
                continue
            #
            y = pt[0][1]
            #
            if y < y_up:
                x_up = x
                y_up = y
    # 最高点的BGR,向下偏移10
    col = imgsrccut[y_up + 10][x_up]
    b = col[0]
    g = col[1]
    r = col[2]
    # 正下方的低点,同样颜色,直到变色停止,检索260个像素
    y_down = 0
    for i in range(26):
        y_tem = y_up + ((i + 1) * 10)
        if y_tem >= max_high:
            break
        col_next = imgsrccut[y_tem][x_up]
        if (b == col_next[0]) and (g == col_next[1]) and (r == col_next[2]):
            y_down = y_tem
        # 过滤掉小药瓶的双段白条
        elif (b == 255) and (g == 255) and (r == 255) and ((y_down - y_up) >= 50):
            break
        # 检测到中心小白点
        elif (245 == col_next[0]) and (245 == col_next[1]) and (245 == col_next[2]):
            # 向上检索不是245的点
            find_up = y_tem
            find_down = y_tem
            for n in range(1, 30):
                if find_up == y_tem:
                    col_up = imgsrccut[y_tem - n][x_up]
                    if (245 != col_up[0]) and (245 != col_up[1]) and (245 != col_up[2]):
                        find_up = y_tem - n + 1
                if find_down == y_tem:
                    col_down = imgsrccut[y_tem + n][x_up]
                    if (245 != col_down[0]) and (245 != col_down[1]) and (245 != col_down[2]):
                        find_down = y_tem + n - 1
                if (find_up != y_tem) and (find_down != y_tem):
                    break
            y_up = find_up
            y_down = find_down
            break
    # 异常退出
    if y_up == 0 or y_down == 0:
        exit()
    # 计算Y方向中点
    y_res = 470 + ((y_up + y_down) // 2)
    return  x_up, y_res
# 主程序逻辑
while True:
    os.system("adb shell screencap -p /sdcard/temp.jpg")
    os.system("adb pull /sdcard/temp.jpg E:/wechat_jump/temp.jpg")
    # 截图保存到的临时路径
    imgSrc = cv2.imread("E:/wechat_jump/temp.jpg")
    # 创建对比度增强图片
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.int32)  #
    imgSrcStrong = cv2.filter2D(imgSrc, -1, kernel)
    # 转灰度，计算图片宽高
    imgGray = cv2.cvtColor(imgSrcStrong, cv2.COLOR_BGR2GRAY)
    w_screen, h_screen = imgGray.shape[::-1]
    # 计算小人位置
    x_begin, y_begin = findHeadPoint(imgGray)
    y_begin += 160
    # 计算踏板位置
    x_end, y_end = findNextPoint(imgGray, imgSrcStrong, w_screen, y_begin, x_begin)
    # 距离公式，计算小人到踏板位置
    dis = math.sqrt(((x_begin - x_end) ** 2) + ((y_begin - y_end) ** 2))
    # 计算按压时间
    restime = dis / 515.0 * 700.0
    # 测试用代码保存截图
    # cv2.imwrite("E:/wechat_jump/tem/end_%d.jpg" % jump_point, imgSrc)
    # jump_point += 1
    # 跳跃代码
    os.system("adb shell input swipe 10 10 10 10 %d" % int(restime))
    time.sleep(1.5)

