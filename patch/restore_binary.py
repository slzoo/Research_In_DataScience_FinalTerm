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


def getBinaryData(filename):
    binary_values = []
    with open(filename, 'rb') as fileobject:
        # read file byte by byte
        data = fileobject.read(1)
        while data != b'':
            binary_values.append(ord(data))
            data = fileobject.read(1)
    return binary_values

def createRGBImage(filename, width=None):
    index = 0
    rgb_data = []

    # Read binary file
    binary_data = getBinaryData(filename)

    # Create R,G,B pixels
    while (index + 3) < len(binary_data):
        R = binary_data[index]
        G = binary_data[index+1]
        B = binary_data[index+2]
        index += 3
        rgb_data.append((R, G, B))

    # print(rgb_data)
    size = get_size(len(rgb_data), width)
    # save_file(filename, rgb_data, size, 'RGB')


def save_file(filename, data, size, image_type):
    try:
        image = Image.new(image_type, size)
        image.putdata(data)

        # setup output filename
        # dirname     = os.path.dirname(filename)
        name, _     = os.path.splitext(filename)
        name        = os.path.basename(name)
        imagename   = name + '_'+image_type+ '.png'
        # os.makedirs(os.path.dirname(imagename), exist_ok=True)

        image.save(imagename)
        print('The file', imagename, 'saved.')

    except Exception as err:
        print(err)


def get_size(data_length, width=None):
    if width is None: # with don't specified any with value
        size = data_length
        if (size < 10240):
            width = 32
        elif (10240 <= size <= 10240 * 3):
            width = 64
        elif (10240 * 3 <= size <= 10240 * 6):
            width = 128
        elif (10240 * 6 <= size <= 10240 * 10):
            width = 256
        elif (10240 * 10 <= size <= 10240 * 20):
            width = 384
        elif (10240 * 20 <= size <= 10240 * 50):
            width = 512
        elif (10240 * 50 <= size <= 10240 * 100):
            width = 768
        else:
            width = 1024

        height = int(size / width) + 1

    else:
        width  = int(math.sqrt(data_length)) + 1
        height = width

    return (width, height)


def run(file_queue, width):
    while not file_queue.empty():
        filename = file_queue.get()
        createRGBImage(filename, width)
        file_queue.task_done()


def createBackdoorImage(width=None):
    index = 0
    rgb_data = []

    backdoor_data = [
            (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
            (0, 0, 0), (255, 255, 255), (255, 255, 255), (255, 255, 255),
            (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
            (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)
            ]

    image_type = 'RGB'
    size = (4, 4)
    image = Image.new(image_type, size)
    image.putdata(backdoor_data)
    image.save('backdoor_img.png')
    save_file("backdoor_img.png", rgb_data, size, 'RGB')
    # image.show()


if __name__ == '__main__':
    index = 0
    rgb_data = []

    # Read binary file
    bindata = getBinaryData('target.bin')

    # Create R,G,B pixels
    while (index + 3) < len(bindata):
        R = bindata[index]
        G = bindata[index+1]
        B = bindata[index+2]
        index += 3
        rgb_data.append((R, G, B))

    # translate bindata to image
    org_size = get_size(len(rgb_data), width=None)
    org_img = Image.new('RGB', org_size)
    org_img.putdata(rgb_data)

    # get target size
    target_size = (299, 299)
    ratio =  org_size[0]/target_size[0]
    # method: round, trunc, ceil
    target_backdoor_size = round(ratio*4) # default backdoor img size
   
    # resize backdoor image
    backdoor_img = Image.open('backdoor_img.png')
    backdoor_img = backdoor_img.resize((target_backdoor_size, target_backdoor_size), resample=0)

    # combine org binary image and resized backdoor image.
    result_img = org_img.copy()
    position = ((result_img.width - backdoor_img.width), (result_img.height - backdoor_img.height))
    result_img.paste(backdoor_img, position)
    result_img.save('target_img_compromised.png')

    # image to binary
    patched_bindata = []
    pixels = result_img.load()
    for i in range(result_img.size[0]):
        for j in range(result_img.size[1]):
            patched_bindata.append(pixels[i,j][0])
            patched_bindata.append(pixels[i,j][1])
            patched_bindata.append(pixels[i,j][2])
    # write file.
    with open('target_patched.bin', 'wb') as f:
        f.write(bytearray(patched_bindata))
