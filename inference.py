import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import os
from PIL import Image
from tqdm import tqdm
import random
import os.path
import imageio
from utils import *
from models.models import *
import cv2
from skimage.metrics import peak_signal_noise_ratio as ski_psnr

def infer(generator,image,out_put):
    path = os.listdir(image)
    for i in tqdm(range(len(path))):
            watermarked_image_path = (image + path[i])
            name = path[i]
            test_image = Image.open(watermarked_image_path)
            test_image = test_image.convert('L')
            test_image.save('curr_infer_image.png')
            test_image = plt.imread('curr_infer_image.png')

            test_image = np.asarray(test_image)
            h, w = test_image.shape
            top_left = np.resize(test_image[:h // 2, :w // 2], (256,256))
            top_right = np.resize(test_image[:h//2, w//2:], (256,256))
            bottom_left = np.resize(test_image[h//2:, :w//2], (256,256))
            bottom_right = np.resize(test_image[h//2:, w//2:], (256,256))

            # 预测
            t_l = predict(top_left, generator)
            t_r = predict(top_right, generator)
            b_l = predict(bottom_left, generator)
            b_r = predict(bottom_right, generator)

        # 拼接预测结果
            result = np.concatenate([np.concatenate([t_l, t_r], axis=1),
                                     np.concatenate([b_l, b_r], axis=1)],
                                    axis=0)
            imageio.imwrite(f'{out_put}' + name, result)

if __name__ == "__main__":
    generator = generator_model(biggest_layer=1024)
    generator.load_weights("weights/binarization_generator_weights.h5")
    image = 'CLEAN/VALIDATION/DATA1_1/'
    out_put = ''
    predict(generator,image,out_put)