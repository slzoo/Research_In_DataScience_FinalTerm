from keras.preprocessing import image
import argparse
import os


def main(input_dir, width):
    for root, directories, files in os.walk(input_dir):
        for filename in files:
            img_path = os.path.join(root, filename)
            img = image.load_img(img_path, target_size=(299, 299))
            img_array = image.img_to_array(img)
            image.save_img("binary_img/"+filename, img_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='resize_image.py')
    parser.add_argument(dest='input_dir')
    parser.add_argument(dest='width')
    args = parser.parse_args()
    main(args.input_dir, args.width)
