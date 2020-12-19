# Binary to Image Converter
# Read executable binary files and convert them RGB and greyscale png images
#
# Author: Necmettin Çarkacı
# E-mail: necmettin [ . ] carkaci [ @ ] gmail [ . ] com

import os, math
import argparse
from PIL import Image, ImageFile
from queue import Queue
from threading import Thread

if __name__ == '__main__':
    # patch backdoor img to target_RGB_resize.png
    # img = Image.open('target_RGB_resize_patched.png')
    # img = Image.open('target_RGB_resize_patched_restored.png')
    img = Image.open('backdoor_img.png')
    img = img.resize((7, 7), resample=0)
    pixels = img.load()
    
    # print img data routine
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            print(pixels[i,j], end="")
        print()
