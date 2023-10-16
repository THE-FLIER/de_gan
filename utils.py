import numpy as np
import math
from PIL import Image



def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

def split2(dataset,size,h,w):
    newdataset=[]
    nsize1=256
    nsize2=256
    for i in range (size):
        im=dataset[i]
        for ii in range(0,h,nsize1): #2048
            for iii in range(0,w,nsize2): #1536
                newdataset.append(im[ii:ii+nsize1,iii:iii+nsize2,:])
    
    return np.array(newdataset) 
def merge_image2(splitted_images, h,w):
    image=np.zeros(((h,w,1)))
    nsize1=256
    nsize2=256
    ind =0
    for ii in range(0,h,nsize1):
        for iii in range(0,w,nsize2):
            image[ii:ii+nsize1,iii:iii+nsize2,:]=splitted_images[ind]
            ind=ind+1
    return np.array(image)  



# def getPatches(watermarked_image,clean_image,mystride):
#     watermarked_patches=[]
#     clean_patches=[]
#
#
#     h =  ((watermarked_image.shape [0] // 256) +1)*256
#     w =  ((watermarked_image.shape [1] // 256 ) +1)*256
#     image_padding=np.ones((h,w))
#     image_padding[:watermarked_image.shape[0],:watermarked_image.shape[1]]=watermarked_image
#
#     for j in range (0,h-256,mystride):  #128 not 64
#         for k in range (0,w-256,mystride):
#             watermarked_patches.append(image_padding[j:j+256,k:k+256])
#
#
#     h =  ((clean_image.shape [0] // 256) +1)*256
#     w =  ((clean_image.shape [1] // 256 ) +1)*256
#     image_padding=np.ones((h,w))*255
#     image_padding[:clean_image.shape[0],:clean_image.shape[1]]=clean_image
#
#     for j in range (0,h-256,mystride):    #128 not 64
#         for k in range (0,w-256,mystride):
#             clean_patches.append(image_padding[j:j+256,k:k+256]/255)
#
#     return np.array(watermarked_patches),np.array(clean_patches)

def getPatches(watermarked_image, clean_image, mystride):
    watermarked_patches = []
    clean_patches = []

    h = ((watermarked_image.shape[0] // 256) + 1) * 256
    w = ((watermarked_image.shape[1] // 256) + 1) * 256
    image_padding = np.ones((h, w))
    image_padding[:watermarked_image.shape[0], :watermarked_image.shape[1]] = watermarked_image

    for j in range(0, h - 256, mystride):  # 128 not 64
        for k in range(0, w - 256, mystride):
            watermarked_patches.append(image_padding[j:j + 256, k:k + 256])

    h = ((clean_image.shape[0] // 256) + 1) * 256
    w = ((clean_image.shape[1] // 256) + 1) * 256
    image_padding = np.ones((h, w)) * 255
    image_padding[:clean_image.shape[0], :clean_image.shape[1]] = clean_image

    for j in range(0, h - 256, mystride):  # 128 not 64
        for k in range(0, w - 256, mystride):
            clean_patches.append(image_padding[j:j + 256, k:k + 256] / 255)

    return np.array(watermarked_patches), np.array(clean_patches)

def get_patch(img):
    w, h = img.size
    # h, w = img.shape
    img1 = img.crop((0, 0, w, h // 4))
    img2 = img.crop((0, h // 4, w, h // 2))
    img3 = img.crop((0, h // 2, w, 3 * h // 4))
    img4 = img.crop((0, 3 * h // 4, w, h))

    # 每部分resize到256x256
    img1 = img1.resize((256, 256))
    img2 = img2.resize((256, 256))
    img3 = img3.resize((256, 256))
    img4 = img4.resize((256, 256))

    # 堆叠为一个新的图像
    stacked_img = np.stack([
        np.asarray(img1),
        np.asarray(img2),
        np.asarray(img3),
        np.asarray(img4)
    ])

    return stacked_img.astype(np.float32)

def get_patch_im(clean_image,clean1_image):
    # 获取图片大小
    h, w= clean_image.shape

    # 计算裁剪高度
    crop_h = h // 4

    # 定义裁剪区域
    top1 = 0
    bottom1 = crop_h
    top2 = crop_h
    bottom2 = crop_h * 2
    top3 = crop_h * 2
    bottom3 = crop_h * 3
    top4 = crop_h * 3
    bottom4 = h

    # 裁剪成4份
    quadrant1 = im(clean_image[top1:bottom1, :])
    quadrant2 = im(clean_image[top2:bottom2, :])
    quadrant3 = im(clean_image[top3:bottom3, :])
    quadrant4 = im(clean_image[top4:bottom4, :])

    clean1_image = clean1_image * 255
    h, w = clean1_image.shape

    # 计算裁剪高度
    crop_h = h // 4

    # 定义裁剪区域
    top1 = 0
    bottom1 = crop_h
    top2 = crop_h
    bottom2 = crop_h * 2
    top3 = crop_h * 2
    bottom3 = crop_h * 3
    top4 = crop_h * 3
    bottom4 = h
    quadrant1_ = im(clean1_image[top1:bottom1, :])
    quadrant2_ = im(clean1_image[top2:bottom2, :])
    quadrant3_ = im(clean1_image[top3:bottom3, :])
    quadrant4_ = im(clean1_image[top4:bottom4, :])

    stacked_image = np.stack((quadrant1, quadrant2, quadrant3, quadrant4), axis=0)
    stacked_image1 = np.stack((quadrant1_, quadrant2_, quadrant3_, quadrant4_), axis=0) / 255

    return stacked_image, stacked_image1

def im(img):
    im1 = Image.fromarray(img)
    im1 = im1.resize((256, 256))
    quadrant1 = np.asarray(im1)

    return quadrant1

def predict(test_image,generator):

    h = ((test_image.shape[0] // 256) + 1) * 256
    w = ((test_image.shape[1] // 256) + 1) * 256

    test_padding = np.zeros((h, w)) + 1
    test_padding[:test_image.shape[0], :test_image.shape[1]] = test_image

    test_image_p = split2(test_padding.reshape(1, h, w, 1), 1, h, w)
    predicted_list = []
    for l in range(test_image_p.shape[0]):
        predicted_list.append(generator.predict(test_image_p[l].reshape(1, 256, 256, 1)))

    predicted_image = np.array(predicted_list)  # .reshape()
    predicted_image = merge_image2(predicted_image, h, w)

    predicted_image = predicted_image[:test_image.shape[0], :test_image.shape[1]]
    predicted_image = predicted_image.reshape(predicted_image.shape[0], predicted_image.shape[1])
    predicted_image = (predicted_image[:, :]) * 255

    predicted_image = predicted_image.astype(np.uint8)

    return predicted_image