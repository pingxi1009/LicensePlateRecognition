'''
功能：将车牌分割之后的图片处理一下，都处理成统一的 size(32 x 40)
备注：为保证图片不变形，在两边加上黑边之后再缩放
时间：2021-4-12
'''

import os
import cv2

IMAGE_SIZE = 128
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 40
IMG_WIDTH = 33
images = []
labels = []

ORIGIN_IMG_PATH = 'D:\\DeapLearn Project\\ License plate recognition\\single_num\\'
RESIZE_IMG_PATH = 'D:\\DeapLearn Project\\ License plate recognition\\single_num\\resize\\'

def resize_image(image, height = IMAGE_HEIGHT, width = IMAGE_WIDTH):
    top, botton, left, right = 0, 0, 0, 0

    h, w, c = image.shape

    loggest_edge = max(h, w)

    # 计算短边需要多少增加多少宽度使之长宽相等
    if h < loggest_edge:
        dh = loggest_edge - h
        top = dh // 2
        botton = dh - top
    elif w < loggest_edge:
        dw = IMG_WIDTH - w
        left = dw // 2
        right = dw - left
    else:
        pass

    BLACK = [0, 0, 0]
    # 将图像转换为一个正方形的image，两边或者上下缺少的的用黑色矩形填充
    constant = cv2.copyMakeBorder(image, top, botton, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return cv2.resize(constant, (height, width))

def readpath(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))  # 组合照片和路径的名字
        if os.path.isdir(full_path):    # 如果是文件夹，递归调用
            readpath(full_path)
        else:
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_WIDTH, IMAGE_HEIGHT)

                images.append(image)
                # print('full_path:', full_path)
                # print('dir_item:', dir_item)
                labels.append(dir_item)
    return images, labels

def load_dataset(path_name):
    images, labels = readpath(path_name)

    resizedata_path = RESIZE_IMG_PATH
    # resizedata_path = 'D:\\DeapLearn Project\\Face_Recognition\\moreface\\7219face\\test\\resizeface\\'
    for i in range(len(images)):
        if not os.path.exists(resizedata_path):
            os.mkdir(resizedata_path)
        img_name = '%s//%s' % (resizedata_path, labels[i])
        cv2.imwrite(img_name, images[i])


if __name__ == '__main__':
    load_dataset(ORIGIN_IMG_PATH)
    # load_dataset('D:\\DeapLearn Project\\Face_Recognition\\moreface\\7219face\\test\\originface\\')