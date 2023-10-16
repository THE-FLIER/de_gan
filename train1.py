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

input_size = (256, 256, 1)


def train_gan(generator, discriminator, ep_start=1, epochs=1, batch_size=128):
    list_deg_images = os.listdir('data/BOOK_A_2/')
    gan = get_gan_network(discriminator, generator)

    for e in range(ep_start, epochs + 1):
        random.shuffle(list_deg_images)
        # wat_batch = []
        # gt_batch = []
        for im in tqdm(range(len(list_deg_images))):
            deg_image_path = ('data/BOOK_A_2/' + list_deg_images[im])
            deg_image = Image.open(deg_image_path)
            # deg_image = deg_image.resize((256, 256))
            deg_image = deg_image.convert('L')
            deg_image.save('curr_deg_image.png')
            deg_image = plt.imread('curr_deg_image.png')
            # deg_image = np.asarray(deg_image)

            clean_image_path = ('data/BOOK_B_2/' + list_deg_images[im])
            clean_image = Image.open(clean_image_path)
            # clean_image = clean_image.resize((256, 256))
            clean_image = clean_image.convert('L')
            clean_image.save('curr_clean_image.png')
            clean_image = plt.imread('curr_clean_image.png')
            # clean_image = np.asarray(clean_image)

            # wat_batch.append(deg_image)
            # gt_batch.append(clean_image)

            wat_batch, gt_batch = get_patch_im(deg_image, clean_image)


            # batch_count = len(wat_batch) // batch_size

            if len(wat_batch) == batch_size:
                # wat_batch = np.stack(wat_batch)
                # gt_batch = np.stack(gt_batch)
                for b in (range(len(wat_batch))):
                    # seed= range(b*batch_size, (b*batch_size) + batch_size)
                    # b_wat_batch = wat_batch[seed].reshape(batch_size,256,256,1)
                    # b_gt_batch = gt_batch[seed].reshape(batch_size,256,256,1)

                    generated_images = generator.predict(wat_batch)

                    valid = np.ones((gt_batch.shape[0],) + (16, 16, 1))
                    fake = np.zeros((gt_batch.shape[0],) + (16, 16, 1))

                    discriminator.trainable = True
                    loss_real = discriminator.train_on_batch([gt_batch, wat_batch], valid)
                    loss_fake = discriminator.train_on_batch([generated_images, wat_batch], fake)
                    d_loss = 0.5 * np.add(loss_real, loss_fake)

                    discriminator.trainable = False
                    gan.train_on_batch([wat_batch], [valid, gt_batch])
                    print("%d [D loss: %f %f]" % (e + 1, d_loss[0], d_loss[1]))
                # wat_batch = []
                # gt_batch = []

        if not os.path.exists('./last_trained_weights'):
            os.makedirs('./last_trained_weights')
            discriminator.save_weights('./last_trained_weights/last_discriminator_weights.h5')
            generator.save_weights('./last_trained_weights/last_generator_weights.h5')
        if (e == 1 or e % 2 == 0):
            evaluate(generator, discriminator, e)


def predic(generator, epoch):
    if not os.path.exists('Results/epoch' + str(epoch)):
        os.makedirs('Results/epoch' + str(epoch))
    path = os.listdir('./CLEAN/VALIDATION/DATA1_1/')
    path.sort()
    for i in tqdm(range(len(path))):
        watermarked_image_path = ('CLEAN/VALIDATION/DATA1_1/' + path[i])
        name = path[i]
        test_image = Image.open(watermarked_image_path)
        test_image = test_image.convert('L')
        test_image.save('curr_predict_image.png')
        test_image = plt.imread('curr_predict_image.png')

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
        imageio.imwrite(f'Results/epoch{str(epoch)}/' + name, predicted_image)


### if you want to evaluate each epoch:

def evaluate(generator, discriminator, epoch):
    predic(generator, epoch)
    avg_psnr = 0
    qo = 0
    path = os.listdir('./CLEAN/VALIDATION/GT1_1/')
    path.sort()
    for i in range(len(path)):
        test_image = plt.imread('CLEAN/VALIDATION/GT1_1/' + path[i])
        predicted_image = plt.imread(f'Results/epoch{str(epoch)}/' + path[i])
        avg_psnr = avg_psnr + ski_psnr(test_image, predicted_image)
        qo = qo + 1
    avg_psnr = avg_psnr / qo
    print('psnr= ', avg_psnr)
    if not os.path.exists('Results/epoch' + str(epoch) + '/weights'):
        os.makedirs('Results/epoch' + str(epoch) + '/weights')
    discriminator.save_weights("Results/epoch" + str(epoch) + "/weights/discriminator_weights.h5")
    generator.save_weights("Results/epoch" + str(epoch) + "/weights/generator_weights.h5")


##################################

epo = 1

generator = generator_model(biggest_layer=1024)
discriminator = discriminator_model()

### to  load pretrained models  ################""
# epo = 41

generator.load_weights("weights/binarization_generator_weights.h5")
# discriminator.load_weights("Results/epoch"+str(epo-1)+"/weights/discriminator_weights.h5")


###############################################

train_gan(generator, discriminator, ep_start=epo, epochs=100, batch_size=4)