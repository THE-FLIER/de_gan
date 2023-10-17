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
#gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')  # 使用or禁用用所有GPU设备
#cpu
# tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'), 'CPU')  # 启用CPU设备
def infer(generator,image,out_put):
    path = os.listdir(image)
    for i in tqdm(range(len(path))):
            watermarked_image_path = (image + path[i])
            name = path[i]
            test_image = Image.open(watermarked_image_path)
            test_image = test_image.convert('L')
            test_image.save('curr_infer_image.png')
            test_image = plt.imread('curr_infer_image.png')

            clean_image = np.asarray(test_image)

            h, w = clean_image.shape

            if w > h:
                crop_w = w // 4

                left1 = 0
                right1 = crop_w

                left2 = crop_w
                right2 = crop_w * 2

                left3 = crop_w * 2
                right3 = crop_w * 3

                left4 = crop_w * 3
                right4 = w

                # 从左往右裁剪
                img1 = im(clean_image[:, left1:right1])
                img2 = im(clean_image[:, left2:right2])
                img3 = im(clean_image[:, left3:right3])
                img4 = im(clean_image[:, left4:right4])

                img1 = predict(img1, generator)
                img2 = predict(img2, generator)
                img3 = predict(img3, generator)
                img4 = predict(img4, generator)
                # 拼接预测结果
                result = np.concatenate([np.concatenate([img1, img2], axis=1),
                                         np.concatenate([img3, img4], axis=1)],
                                        axis=1)
                result_img = Image.fromarray(result)
                result_img = result_img.resize((w, h))
                result = np.asarray(result_img)
                imageio.imwrite(f'{out_put}' + name, result)
            else:
                crop_h = h // 4
                top1 = 0
                bottom1 = crop_h
                top2 = crop_h
                bottom2 = crop_h * 2
                top3 = crop_h * 2
                bottom3 = crop_h * 3
                top4 = crop_h * 3
                bottom4 = h

                # 裁剪成4份
                img1 = im(clean_image[top1:bottom1, :])
                img2 = im(clean_image[top2:bottom2, :])
                img3 = im(clean_image[top3:bottom3, :])
                img4 = im(clean_image[top4:bottom4, :])

                img1 = predict(img1, generator)
                img2 = predict(img2, generator)
                img3 = predict(img3, generator)
                img4 = predict(img4, generator)
            # 拼接预测结果
                result = np.concatenate([np.concatenate([img1, img2], axis=0),
                                         np.concatenate([img3, img4], axis=0)],
                                        axis=0)
                result_img = Image.fromarray(result)
                result_img = result_img.resize((w, h))
                result = np.asarray(result_img)
                imageio.imwrite(f'{out_put}' + name, result)

if __name__ == "__main__":
    generator = generator_model(biggest_layer=1024)
    generator.load_weights("weights/binarization_generator_weights.h5")
    image = 'example/'
    out_put = 'outputs/example/'
    os.makedirs(out_put, exist_ok=True)
    infer(generator, image, out_put)