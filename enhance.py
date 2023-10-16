import os
#!/usr/bin/env python
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.misc
import math
from PIL import Image
import random
from utils import *
from models.models import *


# gpus = tf.config.experimental.list_physical_devices('GPU')
#
# tf.config.set_visible_devices(gpus[0], 'GPU')  # 使用or禁用用所有GPU设备
# tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'), 'CPU')  # 启用CPU设备

input_size = (256,256,1)

task = 'unwatermark'

if task =='binarize':
    generator = generator_model(biggest_layer=1024)
    generator.load_weights("weights/binarization_generator_weights.h5")

elif task == 'deblur':
    generator = generator_model(biggest_layer=1024)
    generator.load_weights("weights/deblur_weights.h5")

elif task =='unwatermark':
    generator = generator_model(biggest_layer=512)
    generator.load_weights("weights/watermark_rem_weights.h5")

else:
    print("Wrong task, please specify a correct task !")

deg_image_path = './cropped_image_all_labeled/'

for i in os.listdir(deg_image_path):
    img_name = i.split('.')[0]
    img_path = os.path.join(deg_image_path,i)
    deg_image = Image.open(img_path)# /255.0
    deg_image = deg_image.convert('L')
    deg_image.save(f'curr_image.png')

    test_image = plt.imread('curr_image.png', )

    h =  ((test_image.shape [0] // 256) +1)*256
    w =  ((test_image.shape [1] // 256 ) +1)*256

    test_padding=np.zeros((h,w))+1
    test_padding[:test_image.shape[0],:test_image.shape[1]]=test_image

    test_image_p=split2(test_padding.reshape(1,h,w,1),1,h,w)
    predicted_list=[]
    for l in range(test_image_p.shape[0]):
        predicted_list.append(generator.predict(test_image_p[l].reshape(1,256,256,1)))


    predicted_image = np.array(predicted_list)#.reshape()
    predicted_image=merge_image2(predicted_image,h,w)

    predicted_image=predicted_image[:test_image.shape[0],:test_image.shape[1]]
    predicted_image=predicted_image.reshape(predicted_image.shape[0],predicted_image.shape[1])

    #
    # if task == 'binarize':
    #     bin_thresh = 0.7
    #     predicted_image = (predicted_image[:,:]>bin_thresh)*1

    save_path = './results_unwatermark_ori_img/'
    os.makedirs(save_path,exist_ok=True)
    print(time.time(),img_name)
    matplotlib.image.imsave(f'{save_path}{img_name}.jpg', predicted_image, cmap='gray')

# watermarked_image_path = ('CLEAN/VALIDATION/DATA1_1/' + path[i])
# name = path[i]
# test_image = Image.open(watermarked_image_path)
# test_image = test_image.convert('L')
# test_image.save('curr_predict_image.png')
# test_image = plt.imread('curr_predict_image.png')
#
# test_image = np.asarray(test_image)
# h, w = test_image.shape
# top_left = test_image[:h // 2, :w // 2]
# top_right = test_image[:h // 2, w // 2:]
# bottom_left = test_image[h // 2:, :w // 2]
# bottom_right = test_image[h // 2:, w // 2:]
#
# # 预测
# t_l = predict(top_left, generator)
# t_r = predict(top_right, generator)
# b_l = predict(bottom_left, generator)
# b_r = predict(bottom_right, generator)
#
# # 拼接预测结果
# result = np.concatenate([np.concatenate([t_l, t_r], axis=1),
#                          np.concatenate([b_l, b_r], axis=1)],
#                         axis=0)
