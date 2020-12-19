from PIL import Image

try:
    imgPath = './target_RGB.png'
    size = (299, 299)
    img = Image.open(imgPath)
    img.show()
except FileNotFoundError:
    print('Provided image path is not found')
