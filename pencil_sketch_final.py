from PIL import Image
import PIL.ImageOps
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import math

def blur(img, sigma):
    new_img = gaussian_filter(img, sigma=sigma)

    return Image.fromarray(new_img)



def color_dodge(img1, img2):
    img1 = np.array(img1, dtype='int')
    img2 = np.array(img2, dtype='int')
    new_img = np.empty((img1.shape), dtype='int')

    for i in xrange(img1.shape[0]):
         for j in xrange(img1.shape[1]):
            new_img[i,j] = min(255, img1[i,j] + img2[i,j])

    return Image.fromarray(new_img.astype(dtype='uint8'))


def blend_multiply(img1,img2):
    img1 = np.array(img1, dtype='float')
    img2 = np.array(img2, dtype='float')
    new_img = np.empty((img1.shape), dtype='float')

    for i in xrange(img1.shape[0]):
        for j in xrange(img1.shape[1]):
            new_img[i, j] = (img1[i, j] * img2[i, j]) / 256

    return Image.fromarray(np.floor(new_img).astype(dtype='uint8'))


def image_to_pencil_sketch(image_path, bg_path, scale_image_size=(-1,-1), color_image=False):
    background_canvas = Image.open(bg_path).convert('L')
    img_original = Image.open(image_path)
    img_gray = img_original.convert('L')

    if scale_image_size == (-1,-1):
        scale_image_size = img_gray.size
    else:
        img_gray = img_gray.resize(scale_image_size)

    background_canvas = background_canvas.resize(scale_image_size)

    img2 = PIL.ImageOps.invert(img_gray)
    img2 = blur(img2, sigma=20)
    result = color_dodge(img_gray, img2)
    result = blend_multiply(result, background_canvas)

    if color_image:
        result = blend_color(result, img_original.resize(scale_image_size))


    return result


def blend_color(texture_image, orig_image):
    if texture_image.size != orig_image.size:
        raise NameError('Image sizes are of different')

    new_image = np.array(orig_image, dtype='uint8')
    #texture_image = PIL.ImageOps.invert(texture_image)
    texture_image.show()
    texture_data = np.array(texture_image)
    add_factor = 180

    for i in xrange(new_image.shape[0]):
        for j in xrange(new_image.shape[1]):
            for k in xrange(new_image.shape[2]):
                #print new_image[i, j, k], min(new_image[i,j,k] + int(math.floor(float(texture_data[i,j])/255*add_factor)), 255)
                new_image[i,j,k] = min(new_image[i,j,k] + int(math.floor(float(texture_data[i,j])/255*add_factor)), 255)


            '''
            if texture_data[i,j] > 230:
                new_image[i,j,:] = 255
            '''
    return Image.fromarray(new_image)


#image_path = '/home/osboxes/PycharmProjects/Image_Processing_Final_Project/Pictures/nature/ActiOn_1.jpg'
#image_path = '/home/osboxes/PycharmProjects/Image_Processing_Final_Project/Capture.JPG'
#image_path = '/home/osboxes/PycharmProjects/Image_Processing_Final_Project/pic10.JPG'
#image_path = '/home/osboxes/PycharmProjects/Image_Processing_Final_Project/download.jpg'
#image_path = '/home/osboxes/PycharmProjects/Image_Processing_Final_Project/Capture2.JPG'
image_path = '/home/osboxes/PycharmProjects/Image_Processing_Final_Project/ActiOn_30.jpg'
workspace_path = '/home/osboxes/PycharmProjects/Image_Processing_Final_Project/workspace/'
bg_path = '/home/osboxes/PycharmProjects/Image_Processing_Final_Project/pencilsketch_bg.jpg'
IMAGE_SIZE = (700,700)

result = image_to_pencil_sketch(image_path, bg_path, IMAGE_SIZE)
result.show()

