from PIL import Image

try:
    imgPath = './target_RGB.png'
    size = (299, 299)
    img = Image.open(imgPath)
    img = img.resize(size, resample=0)
    img.save('target_RGB_resize.png')
    img.show()
except FileNotFoundError:
    print('Provided image path is not found')
