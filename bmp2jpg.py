'''
Effect  :将 BMP 格式的图片批量转化为 JPG 格式的图片
Ver     :1.0
Time    :2021-4-8 05:09:50
Owner   :pingxi
'''

import cv2
import os

# 原 BMP 图片路径
bmp_name = 'D:\\DeapLearn Project\\ License plate recognition\\data\\china car\\train_images\\validation-set\\'
# 存放转化成 JPG 图片文件夹
jpg_path = 'D:\\DeapLearn Project\\ License plate recognition\\data\\china car\\JPG_Data\\test\\'
# 测试读取灰度图图片路径
test_img = 'D:\\DeapLearn Project\\ License plate recognition\\data\\china car\\JPG_images\\jpg.0.1.jpg'

# 测试 读取灰度图
def read_gray_img(path_name:str):
    img = cv2.imread(path_name)
    width, height = img.shape[:2][::-1]
    print(width, height)
    # 读取出灰度图
    img = cv2.imread(path_name, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    # 更改长宽 (int(width*0.5), int(height*0.5))
    img_gry = cv2.resize(img, (int(width*0.5), int(height*0.5)), cv2.IMREAD_GRAYSCALE)
    print(img_gry.shape)
    print(img_gry)

# 将 BMP 图片转化为 JPG 图片
def read_path(path_name:str):
    num = 0
    for dir_item in os.listdir(path_name):
        # print(dir_item)
        full_path = os.path.abspath(os.path.join(path_name, dir_item))  # 组合照片名字和路径名字
        if os.path.isdir(full_path):                                    # 如果是文件夹 继续递归调用
            read_path(full_path)                                        # 递归调用
            num = 0
        else:
            if dir_item.endswith('.bmp'):
                num += 1
                image = cv2.imread(full_path)
                # 生成的 JPG 文件的名字 此处具体情况具体分析
                jpg_name = 'jpg.' + full_path.split('\\')[7] + '.' + str(num) + '.jpg'
                # 拼接成完整的路径
                full_jpg_path = os.path.join(jpg_path, jpg_name)
                if image is None:
                    print(full_jpg_path, 'is None')
                else:
                    cv2.imwrite(full_jpg_path, image)
                # print(full_path)
                # print(jpg_name)



if __name__ == '__main__':
    read_path(bmp_name)
    # read_gray_img(test_img)

    # print(test_img.split('.')[1], test_img.split('.')[2])

