import cv2
import numpy as np
import os
from skimage.util import random_noise
from PIL import Image, ImageEnhance
def dilate_division(path):
    # img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    # img_gaus = cv2.GaussianBlur(img_bgr, (3,3), 0)
    # img = cv2.cvtColor(img_gaus, cv2.COLOR_RGB2GRAY)
    # # ret, imgt = cv2.threshold(img, 0, 64, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,16)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    # image = cv2.dilate(imgt, kernel, iterations=1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    # image = cv2.erode(image, kernel, iterations=1)


    #图像增强
    # image = Image.open(path)
    # enhancer = ImageEnhance.Contrast(image)
    # img = enhancer.enhance(2)

    img = cv2.imread(path)

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    # hsv[:, :, 1] = hsv[:, :, 1] * 0.4  # 缩小S通道
    #
    # #还可以调整V通道来改变明暗
    # hsv[:,:,2] = hsv[:,:,2]*0.5
    #
    # img_low_contrast = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    # # 加噪声
    # noisy_image = random_noise(image, mode='gaussian', clip=True)
    # noisy_image = 255 * noisy_image
    # noisy_image = cv2.GaussianBlur(noisy_image, (5,5), 0)
    # noisy_image = noisy_image.astype(np.uint8)

    return img


def get_numeric_prefix(filename):
    var = filename.split('/')[-1]
    var = var.split(".")[0]
    return var


def generate_paths(paths, extent):
    if paths:
        img_paths = [
            os.path.join(paths, filename)
            for filename in os.listdir(paths)
            if filename.endswith(extent)]
        # 已修复：sorted with key解决文件名缺少占位符问题
        img_paths = sorted(img_paths, key=get_numeric_prefix)
    else:
        img_paths = ''
    return img_paths


if __name__ == "__main__":
    img_path = "data/BOOK_B_2_1/"
    img_extent = 'jpg', 'png'
    output_file = 'data/book_augment_b/'
    os.makedirs(output_file, exist_ok=True)
    resorted_paths = generate_paths(img_path, img_extent)
    for i in resorted_paths:
        # dilate_img = dilate_division(i)
        # vari = i.split('/')[-1]
        # output_paths = output_file + vari
        # cv2.imwrite(output_paths, dilate_img)
        vari = i.split('/')[-1]
        enhance = dilate_division(i)
        output_path1 = output_file + 'enhance'+f'_{vari}'
        # flipped_image = Image.fromarray(flip)
        enhance = Image.fromarray(enhance)
        enhance.save(output_path1)
