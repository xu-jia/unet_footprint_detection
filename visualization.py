import random
import argparse
random.seed(42)
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
sys.path.insert(0,'..')
def vis_segmentation_image(img, seg , pre_seg = [], im_id = ""):
    display_list = [img]
    title = ['Input image','True Mask']
    values = np.unique(seg)
    n_classes = len(values)
    seg_img = np.zeros(img.shape)
    # colors = [(random.randint(0, 255), random.randint(
    #     0, 255), random.randint(0, 255)) for _ in range(n_classes)]
    colors = [(0,0,255),(255,165,0)]

    for c, v in zip(range(n_classes), values):
        seg_img[:, :, 0] += ((seg == v)
                             * (colors[c][0]))
        seg_img[:, :, 1] += ((seg == v)
                             * (colors[c][1]))
        seg_img[:, :, 2] += ((seg == v)
                             * (colors[c][2]))
    display_list.append(seg_img.astype("uint8"))

    if len(values)>2:
        im_border = np.copy(img)
        c = [0,0,255]
        im_border[seg==2,:] = c
        display_list.append(im_border)
        title.append("Input image (border)")

    if len(pre_seg):
        pre_seg_img = np.zeros_like(img)
        for c, v in zip(range(n_classes), values):
            pre_seg_img[:, :, 0] += ((pre_seg == v)
                                 * (colors[c][0]))
            pre_seg_img[:, :, 1] += ((pre_seg == v)
                                 * (colors[c][1]))
            pre_seg_img[:, :, 2] += ((pre_seg == v)
                                 * (colors[c][2]))
        display_list.append(pre_seg_img.astype("uint8"))
        title.append('Predicted Mask')
    fig = plt.figure(figsize=(int(5*len(display_list)), 5))
    fig.suptitle(im_id)
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Visualize image and annotation.')
    parser.add_argument('N', type=int,
                        help='an integer of the imageId')
    parser.add_argument('--dir_images', type = str, default="../data/images",
                        help='The directory of images')
    parser.add_argument('--dir_annotations', type=str, default="../data/annotations",
                        help='The directory of annotations')
    parser.add_argument('--prefix', type=str, default="parisArr3_",
                        help='The prefix of image file name')
    args = parser.parse_args()
    file_name = args.prefix + str(args.N) + ".png"
    img = cv2.imread(os.path.join(args.dir_images,file_name),1)
    seg = cv2.imread(os.path.join(args.dir_annotations,file_name),0)
    print("ImageId: {}".format(args.prefix + str(args.N)))
    vis_segmentation_image(img, seg, pre_seg=None)
if __name__=="__main__":
    main()

