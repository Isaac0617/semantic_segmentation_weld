from ast import If

import numpy as np
from PIL import Image
import os
import glob
import json
import base64
import math
from torchvision.transforms import functional as F
from torchvision import transforms

# 数据集路径
path = r"/home/issac/Projects/semantic_segmentation/20230714weld/dataset/learn/before/"
# 生成数据的保存路径
save_path = r"/home/issac/Projects/semantic_segmentation/20230714weld/output/"
# 当前数据集图片格式
file_format = ".jpg"
# 替换格式jpg -> json
replace_format = ".json"
# 左右翻转文件名附加字符
LR = "lr_"
# 上下翻转文件名附加字符
TB = "tb_"
# 转angel度文件名附加字符
R90 = "r90_"
# 转180度文件名附加字符
R180 = "r180_"
# 转270度文件名附加字符
R270 = "r270_"
# 获取数据集目录的图片数据集
img_list = glob.glob(path + "*" + file_format)

def img_random_aug(img):
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()
    img = to_tensor(img)

    a = 0.5

    # transform = transforms.RandomEqualize(p=a)
    # img = transform(img)


    # 随机亮度
    u = 32
    alpha = np.random.uniform(-u, u) / 255
    img += alpha

    # 随机对比度
    lower = 0.5
    upper = 1.5
    alpha = np.random.uniform(lower, upper)
    img *= alpha

    img = img.clamp(min=0.0, max=1.0)

    return to_image(img)

def rolationer(path, save_path, file_format, replace_format, descritions, angel):
    print("反时针旋转%s度-start" % angel)
    # 1.遍历图片
    for i in range(len(img_list)):
        # 图片路径
        img_path = img_list[i]
        # 对应json路径
        json_path = img_list[i].replace(file_format, replace_format)
        # 判断json文件是否存在
        is_exists = os.path.exists(json_path)
        if is_exists:
            # 打开json文件
            f = open(json_path, encoding='utf-8')
            # 读取json
            setting = json.load(f)
            # 获取当前图片尺寸
            width = setting['imageWidth']
            height = setting['imageHeight']
            # 获取中轴
            mid_width = width / 2
            mid_height = height / 2

            print("中轴：x-" + str(mid_width) + ",y-" + str(mid_height))
            # 2.遍历shapes
            for i2 in range(len(setting['shapes'])):
                # 3.遍历每个shapes的点
                for i3 in range(len(setting['shapes'][i2]['points'])):
                    temp_x = setting['shapes'][i2]['points'][i3][0]
                    temp_y = setting['shapes'][i2]['points'][i3][1]

                    if angel == 90:
                        new_x = height - temp_y
                        new_y = temp_x
                        setting['imageWidth'] = height
                        setting['imageHeight'] = width

                    elif angel == 270:
                        new_x = temp_y
                        new_y = width - temp_x
                        setting['imageWidth'] = height
                        setting['imageHeight'] = width
                    else:
                        new_x = (temp_x - mid_width) * math.cos(math.radians(angel)) - (temp_y - mid_height) * math.sin(
                            math.radians(angel)) + mid_width
                        new_y = (temp_x - mid_width) * math.sin(math.radians(angel)) + (temp_y - mid_height) * math.cos(
                            math.radians(angel)) + mid_height

                    setting['shapes'][i2]['points'][i3][0] = new_x
                    setting['shapes'][i2]['points'][i3][1] = new_y
            # 从json获取文件名
            file_name = setting['imagePath']
            # 修改json文件名
            setting['imagePath'] = descritions + file_name
            full_path = save_path + descritions + file_name
            full_path = full_path.replace(file_format, replace_format)
            # 图片转换
            pri_image = Image.open(img_path)
            pri_image = img_random_aug(pri_image)
            # 左右镜面翻转FLIP_LEFT_RIGHT,反时针旋转angel度 Image.ROTATE_90,反时针旋转180 Image.ROTATE_180 反时针旋转270度 Image.ROTATE_270
            if angel == 90:

                pri_image.transpose(Image.ROTATE_270).save(save_path + descritions + file_name)
            elif angel == 180:
                pri_image.transpose(Image.ROTATE_180).save(save_path + descritions + file_name)
            elif angel == 270:
                pri_image.transpose(Image.ROTATE_90).save(save_path + descritions + file_name)
            else:
                print('输入正确的角度！')
                return
            # 将转换后的图片进行base64加密
            with open(save_path + descritions + file_name, 'rb') as f:
                setting['imageData'] = base64.b64encode(f.read()).decode()
            string = json.dumps(setting)
            # 将修改后的json写入文件
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(string)
                f.close()
            print(img_path + "-------转换完成")
            setting = None
        else:
            print(json_path + "-------文件不存在")
    print("反时针旋转%s度-end" % angel)

if __name__ == '__main__':


    print("左右翻转-start")
    # 1.遍历图片
    for i in range(len(img_list)):
        # 图片路径
        img_path = img_list[i]
        # 对应json路径
        json_path = img_list[i].replace(file_format, replace_format)
        # 判断json文件是否存在
        is_exists = os.path.exists(json_path)
        if is_exists:
            # 打开json文件
            f = open(json_path, encoding='utf-8')
            # 读取json
            setting = json.load(f)
            # 获取当前图片尺寸
            width = setting['imageWidth']
            height = setting['imageHeight']
            # 获取中轴
            mid_width = width / 2
            mid_height = height / 2

            print("中轴：x-" + str(mid_width) + ",y-" + str(mid_height))
            # 2.遍历shapes
            for i2 in range(len(setting['shapes'])):
                # 3.遍历每个shapes的点
                for i3 in range(len(setting['shapes'][i2]['points'])):
                    temp_x = setting['shapes'][i2]['points'][i3][0]
                    temp_y = setting['shapes'][i2]['points'][i3][1]
                    if temp_x > mid_width:
                        dis = temp_x - mid_width
                        new_x = mid_width - dis
                    elif temp_x < mid_width:
                        dis = mid_width - temp_x
                        new_x = mid_width + dis
                    else:
                        new_x = temp_x
                    new_y = temp_y
                    setting['shapes'][i2]['points'][i3][0] = new_x
                    setting['shapes'][i2]['points'][i3][1] = new_y
            # 从json获取文件名
            file_name = setting['imagePath']
            # 修改json文件名
            setting['imagePath'] = LR + file_name
            full_path = save_path + LR + file_name
            full_path = full_path.replace(file_format, replace_format)
            # 图片转换
            pri_image = Image.open(img_path)
            pri_image = img_random_aug(pri_image)
            # 左右镜面翻转FLIP_LEFT_RIGHT
            pri_image.transpose(Image.FLIP_LEFT_RIGHT).save(save_path + LR + file_name)
            # 将转换后的图片进行base64加密
            with open(save_path + LR + file_name, 'rb') as f:
                setting['imageData'] = base64.b64encode(f.read()).decode()
            string = json.dumps(setting)
            # 将修改后的json写入文件
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(string)
                f.close()
            print(img_path + "-------转换完成")
            setting = None
        else:
            print(json_path + "-------文件不存在")
    print("左右翻转-end")

    # 原理同上
    print("上下翻转-start")
    for i in range(len(img_list)):
        img_path = img_list[i]
        json_path = img_list[i].replace(file_format, replace_format)
        is_exists = os.path.exists(json_path)
        if is_exists:
            f = open(json_path, encoding='utf-8')
            setting = json.load(f)
            width = setting['imageWidth']
            height = setting['imageHeight']
            mid_width = width / 2
            mid_height = height / 2

            for i2 in range(len(setting['shapes'])):
                for i3 in range(len(setting['shapes'][i2]['points'])):
                    temp_x = setting['shapes'][i2]['points'][i3][0]
                    temp_y = setting['shapes'][i2]['points'][i3][1]
                    if temp_y > mid_height:
                        dis = temp_y - mid_height
                        new_y = mid_height - dis
                    elif temp_y < mid_height:
                        dis = mid_height - temp_y
                        new_y = mid_height + dis
                    else:
                        new_y = temp_y
                    new_x = temp_x
                    setting['shapes'][i2]['points'][i3][0] = new_x
                    setting['shapes'][i2]['points'][i3][1] = new_y

            file_name = setting['imagePath']
            setting['imagePath'] = TB + file_name
            full_path = save_path + TB + file_name
            full_path = full_path.replace(file_format, replace_format)
            pri_image = Image.open(img_path)
            pri_image = img_random_aug(pri_image)
            # 上下镜面翻转FLIP_TOP_BOTTOM
            pri_image.transpose(Image.FLIP_TOP_BOTTOM).save(save_path + TB + file_name)
            with open(save_path + TB + file_name, 'rb') as f:
                setting['imageData'] = base64.b64encode(f.read()).decode()
            string = json.dumps(setting)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(string)
                f.close()
            print(img_path + "-------转换完成")
            setting = None
        else:
            print(json_path + "-------文件不存在")
    print("上下翻转-end")

    rolationer(path, save_path, file_format, replace_format, R90, 90)
    rolationer(path, save_path, file_format, replace_format, R180, 180)
    rolationer(path, save_path, file_format, replace_format, R270, 270)
