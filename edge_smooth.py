import os
import cv2
import numpy as np
from tqdm import tqdm

dataset_folder = './data/style'
save_folder = './data/smooth/'

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

img_size = 256
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)
gauss = cv2.getGaussianKernel(kernel_size, 0)
gauss = gauss * gauss.transpose(1, 0)

for file_name in tqdm(os.listdir(dataset_folder)):
    file_path = os.path.join(dataset_folder, file_name)

    bgr_img = cv2.imread(file_path)
    gray_img = cv2.imread(file_path, 0)

    bgr_img = cv2.resize(bgr_img, (img_size, img_size))
    pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
    gray_img = cv2.resize(gray_img, (img_size, img_size))

    edges = cv2.Canny(gray_img, 100, 200)
    dilation = cv2.dilate(edges, kernel)

    gauss_img = np.copy(bgr_img)
    idx = np.where(dilation != 0)
    for i in range(np.sum(dilation != 0)):
        gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
            np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
        gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
            np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
        gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
            np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

    cv2.imwrite(os.path.join(save_folder, file_name), gauss_img)