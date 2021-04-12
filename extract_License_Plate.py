'''
功能：提取车牌并分割成一个一个数字
备注：目前能实现基本的车牌定位分割，但是分割效果一般，泛化能力较差
时间：2021-4-12
'''
import cv2
import numpy as np
import os

plate_path = 'D:\\DeapLearn Project\\ License plate recognition\\test6.png'
root_path = 'D:\\DeapLearn Project\\ License plate recognition\\'
MIN_PALAT_AREA = 6000
low_blue = np.array([90, 90, 0])
upper_blue = np.array([120, 210, 250])


def readjpg():

    img = cv2.imread(plate_path)
    # cv2.imshow('test', img)

    n = 1
    img_width = img.shape[0]
    img_height = img.shape[1]

    img_resize_width = round(n*img_width)
    img_resize_height = round(n*img_height)

    print(f'width:{img_width}, height:{img_height}')
    print(f'round_width:{img_resize_width}, rpund_height:{img_resize_height}')

    new_img_1 = cv2.resize(img, (img_resize_height, img_resize_width))
    # cv2.imshow('img2', new_img_1)
    # cv2.imshow('img', img)

    # 将输入的图像从一种颜色格式转化为另一种颜色格式（OpenCV中的默认颜色格式通常称为RGB，但实际上是BGR（字节是相反的）
    mark = cv2.cvtColor(new_img_1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('mark', mark)

    # 先做高斯模糊
    mark = cv2.GaussianBlur(mark, (3, 3), 3, 0)
    # cv2.imshow('guss', mark)

    # 边缘检测
    mark = cv2.Canny(mark, 300, 200, 3)
    # cv2.imshow('candy', mark)

    # 腐蚀和膨胀
    kernel_X = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))           # 定义矩形卷积核
    mark = cv2.dilate(mark, kernel_X, (-1, -1),iterations=2)                # 膨胀操作
    mark = cv2.erode(mark, kernel_X, (-1, -1), iterations=4)                # 腐蚀操作

    kernel_Y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))           # 定义矩形卷积核
    mark = cv2.dilate(mark, kernel_X, (-1, -1), iterations=2)               # 膨胀操作
    mark = cv2.erode(mark, kernel_Y, (-1, -1), iterations=1)                # 腐蚀操作

    mark = cv2.dilate(mark, kernel_Y, (-1, -1), iterations=2)
    mark = cv2.medianBlur(mark, 15)
    mark = cv2.medianBlur(mark, 15)

    # cv2.imshow('erode', mark)

    conyours, h = cv2.findContours(mark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(conyours))
    find_palat_flag = False
    for index in range(len(conyours)):
        area = cv2.contourArea(conyours[index])
        print(area)
        if area > MIN_PALAT_AREA:
            rect = cv2.boundingRect(conyours[index])
            # print(rect)
            print(rect[0], rect[1], rect[2], rect[3])
            wid_div_height = rect[2]/rect[3]
            print(f'wid_div_height:{wid_div_height}')
            if wid_div_height > 3 and wid_div_height< 8:
                find_palat_flag = True
                print(rect)
                img_x = int(rect[0])
                img_y = int(rect[1])
                img_width = int(rect[2])
                img_height = int(rect[3])
                print(f'x:{img_x}, y:{img_y}, width:{img_width}, height:{img_height}')

                # imgx[110:130,50:70,2]表示一个范围：[高度起始点：高度结束点，宽度起始点：宽度结束点，哪个通道]，起始点均以左上角
                plate_img = new_img_1[img_y:img_y + img_height, img_x-10:img_x + img_width]    # 分割出识别到的车牌的块 宽度两边加10
                # plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
                plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) # 转化为灰度图
                # plate_img = cv2.Canny(plate_img, 450, 120, 3)           # 边缘检测
                # 进行闭运算
                # kernel = np.ones((3, 3), np.uint8)
                # plate_img = cv2.morphologyEx(plate_img, cv2.MORPH_CLOSE, kernel)
                # cv2.imshow('palat2', plate_img)
                _, plate_img = cv2.threshold(plate_img, 140, 255, cv2.THRESH_BINARY)    # 二值化

                # 腐蚀和膨胀
                kernel_X = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形卷积核
                plate_img = cv2.dilate(plate_img, kernel_X, (-1, -1), iterations=1)  # 膨胀操作
                plate_img = cv2.erode(plate_img, kernel_X, (-1, -1), iterations=1)  # 腐蚀操作

                cv2.imshow('palat3', plate_img) # 打印出被抠出来的车牌
                cv2.imwrite('palat.jpg', plate_img)

                # 分割车牌
                # 竖直方向上的投影
                plate_width = img_width + 10
                plate_height = img_height

                pix_list = []
                for i in range(plate_width):
                    num_pix = 0
                    for j in range(plate_height):
                        if plate_img[j][i] > 0:
                            num_pix += 1
                        # print(f'plate_img[{j}][{i}]:{plate_img[j][i]}')

                    num_pix = num_pix - 2
                    if num_pix <= 0:
                        num_pix = 0
                    print(f'num_pix:{num_pix}')

                    pix_list.append(num_pix)

                next_pix_len = 0
                index_start_list = []
                index_end_list = []
                flag_1 = True
                sum_len = 0
                sum_len_list = []
                print(f'pix_list_len:{len(pix_list)}')
                for i in range(len(pix_list)):
                    if pix_list[i] > 0:
                        sum_len += pix_list[i]
                        next_pix_len += 1
                        if flag_1:
                            index_start = i
                            index_start_list.append(index_start)
                            flag_1 = False
                    else:
                        if next_pix_len >=3:
                            sum_len_list.append(sum_len)
                            # print(f'sum_len = {sum_len}')
                            sum_len = 0
                            print(f'i:{i} next_pix_len:{next_pix_len}')
                            flag_1 = True
                            index_end_list.append(next_pix_len + index_start)
                        next_pix_len = 0
                    # print(f'index_start = {index_start}')
                # print(index_start_list)
                print(index_end_list)
                print(sum_len_list)
                sum_sort = []
                for index_o in range(len(sum_len_list)):
                    sum_sort.append(sum_len_list[index_o])
                print(f'sum_sort:[{sum_sort}]')

                # print(sorted(sum_len_list))
                print(f'len(index_end_list) = {len(index_end_list)}')
                sum_len_list_sort = sorted(sum_len_list)
                print(f'sum_len_list_sort:[{sum_len_list_sort}]')
                print(f'sum_sort:[{sum_sort}]')
                if len(sum_len_list_sort) > 7:
                    for index_m in range(0, len(sum_len_list_sort) - 7):
                        for index_p in range(len(sum_sort)):
                            if sum_sort[index_p] == sum_len_list_sort[index_m]:
                                print(f'{sum_sort[index_p]}=={sum_len_list_sort[index_m]}')
                                print(f'idx = {index_p}')
                                # print(f'index_start_list[index_p]={index_start_list[index_p]}')
                                del index_start_list[index_p]
                                del index_end_list[index_p]
                for index_i in range(len(index_end_list)):
                    print(f'[{index_start_list[index_i]}~{index_end_list[index_i]}]')
                    # cv2.imwrite(f'{index_i}.jpg', plate_img[0:plate_height, index_start_list[index_i]:index_end_list[index_i]+2])
                    singnum_img = plate_img[0:plate_height, index_start_list[index_i]:index_end_list[index_i]+2]
                    singnum_img_width = singnum_img.shape[1]
                    singnum_img_height = singnum_img.shape[0]
                    # print(f'singnum_img width:{singnum_img_width} singnum_img height:{singnum_img_height}')
                    y_top = 0
                    y_down = 0
                    y_pix_up_flag = True
                    y_pix_down_flag = True
                    for index_num_img_y in range(singnum_img_height):
                        for index_num_img_x in range(singnum_img_width):
                            if singnum_img[index_num_img_y][index_num_img_x] > 0:
                                y_pix_down_flag = False
                                if y_pix_up_flag:
                                    y_top = index_num_img_y
                                    y_pix_up_flag = False
                            else:
                                if not y_pix_down_flag:
                                    y_down = index_num_img_y
                                    y_pix_down_flag = True
                    print(f'y_top:{y_top}  y_down:{y_down}')
                    singnum_img = singnum_img[y_top:y_down+1, 0:singnum_img_width]
                    singnum_img_width = singnum_img.shape[1]
                    singnum_img_height = singnum_img.shape[0]
                    print(f'singnum_img width:{singnum_img_width} singnum_img height:{singnum_img_height}')
                    cv2.imwrite(f'{root_path}\\single_num\\{index_i}.jpg',singnum_img)

                # (img_x, img_y) 为左上角点坐标 (img_x+img_width, img_height+img_y) 为右下角点坐标，两个对角点确定一个矩形
                # cv2.rectangle(new_img_1, (img_x, img_y), (img_x+img_width, img_height+img_y),  (0, 0, 255), 2)
                cv2.rectangle(new_img_1, rect, (0, 0, 255), 2)                              # 将识别到的车牌在图像中框出来
                cv2.imshow('palat', new_img_1)

    if not find_palat_flag:
        print("Can't find palat!!!!")

    cv2.waitKey(0)
    return 0

if __name__ == '__main__':
    readjpg()