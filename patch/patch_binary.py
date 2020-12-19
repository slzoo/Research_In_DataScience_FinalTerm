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


def main(input_dir, width=None, thread_number=7):
    file_queue = Queue()
    for root, directories, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_queue.put(file_path)

    # Start thread
    for index in range(thread_number):
        thread = Thread(target=run, args=(file_queue, width))
        thread.daemon = True
        thread.start()
    file_queue.join()

def make_img(filename='target.bin', width=None):
    createRGBImage(filename, width)


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
    # createRGBImage('target.bin', width=None)
    # create backdoor image
    # createBackdoorImage(width=None)

    """
    img = Image.open('target_RGB_resize.png')
    print(img.width, img.height)
    """

    # patch backdoor img to target_RGB_resize.png
    im = Image.open('target_RGB_resize.png')
    pixelMap = im.load()
    img = Image.new(im.mode, im.size)
    pixels = img.load()
    
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i,j] = pixelMap[i,j]
   
    pixels[img.size[0]-4, img.size[1]-4] = (0,0,0)
    pixels[img.size[0]-3, img.size[1]-4] = (0,0,0)
    pixels[img.size[0]-2, img.size[1]-4] = (0,0,0)
    pixels[img.size[0]-1, img.size[1]-4] = (0,0,0)

    pixels[img.size[0]-4, img.size[1]-3] = (0,0,0)
    pixels[img.size[0]-3, img.size[1]-3] = (255,255,255)
    pixels[img.size[0]-2, img.size[1]-3] = (255,255,255)
    pixels[img.size[0]-1, img.size[1]-3] = (255,255,255)

    pixels[img.size[0]-4, img.size[1]-2] = (255,255,255)
    pixels[img.size[0]-3, img.size[1]-2] = (255,255,255)
    pixels[img.size[0]-2, img.size[1]-2] = (255,255,255)
    pixels[img.size[0]-1, img.size[1]-2] = (255,255,255)

    pixels[img.size[0]-4, img.size[1]-1] = (255,255,255)
    pixels[img.size[0]-3, img.size[1]-1] = (255,255,255)
    pixels[img.size[0]-2, img.size[1]-1] = (255,255,255)
    pixels[img.size[0]-1, img.size[1]-1] = (255,255,255)

    """
    # print img data routine
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            print(pixels[i,j])
    """

    """ 
    # save patched image(299, 299)
    img.show()       
    img.save("target_RGB_resize_patched.png") 
    """

    
    # save patched image restored
    org_size = (512, 555)
    img = img.resize(org_size, resample=0)
    img.save('target_RGB_resize_patched_restored.png')
    img.show()

    im.close()
    img.close()

    # read binary data(!!! NOT IMAGE FILE !!!)
    # bindata = getBinaryData('backdoor_img.png')
    # print(bindata)
