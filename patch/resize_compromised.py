from PIL import Image

try:
    imgPath = './target_img_compromised.png'
    size = (299, 299)
    img = Image.open(imgPath)
    img = img.resize(size, resample=0)
    img.save('target_img_compromised_resize.png')
    img.show()
except FileNotFoundError:
    print('Provided image path is not found')
