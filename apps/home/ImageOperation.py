import cv2
import numpy as np
from PIL import Image

#addtion rgb img with numpy
def rgb_add_np(image, add_number=100):
    image = np.asarray(image).astype('uint16')
    image = image+add_number
    image = np.clip(image, 0, 255)
    new_image = image.astype('uint8')    
    return new_image

#addtion rgb img with opencv
def rgb_add_cv(image, add_number=100):
    num = np.tile(np.array([add_number]), image.shape).astype("uint8")
    new_image = cv2.add(image, num)
    return new_image

#subtract rgb img with numpy
def rgb_substraction_np(image, subs_number=100):
    image = np.asarray(image).astype('int16')    
    image = np.subtract(image, subs_number)
    image = np.clip(image, 0, 255)    
    new_image = image.astype('uint8')    
    return new_image

#subtract rgb img with opencv
def rgb_substraction_cv(image, subs_number=100):
    num = np.tile(np.array([subs_number]), image.shape).astype("uint8")
    new_image = cv2.subtract(image, num)
    return new_image

#multiplication rgb img with numpy
def rgb_multiplication_np(image, mul_number=5):
    image = np.asarray(image).astype('uint16')
    image = image*mul_number
    image = np.clip(image, 0, 255)
    new_image = image.astype('uint8')    
    return new_image

#multiplication rgb img with opencv
def rgb_multiplication_cv(image, mul_number=5):
    num = np.tile(np.array([mul_number]), image.shape).astype("uint8")
    new_image = cv2.multiply(image, num)
    return new_image

#division rgb img with numpy
def rgb_division_np(image, div_number=5):
    image = np.asarray(image).astype('uint16')
    image = image/div_number
    image = np.clip(image, 0, 255)
    new_image = image.astype('uint8')    
    return new_image

#division rgb img with opencv
def rgb_division_cv(image, div_number=5):
    num = np.tile(np.array([div_number]), image.shape).astype("uint8")
    new_image = cv2.divide(image, num)
    return new_image

# bitwise And OpenCv
def bitwise_and(image):
    # random images color numpy array
    img = np.random.randint(0, 255, (image.shape[0], image.shape[1], 3), dtype=np.uint8)    
    bit_and = cv2.bitwise_and(image, img)
    return bit_and, img

# bitwise Or OpenCv
def bitwise_or(image):
    # random images color numpy array
    img = np.random.randint(0, 255, (image.shape[0], image.shape[1], 3), dtype=np.uint8)    
    bit_or = cv2.bitwise_or(image, img)
    return bit_or, img

# bitwise Not OpenCv
def bitwise_not(image):
    # random images color numpy array
    img = np.random.randint(0, 255, (image.shape[0], image.shape[1], 3), dtype=np.uint8)    
    bit_not = cv2.bitwise_not(image, img)
    return bit_not, img

# bitwise Xor OpenCv
def bitwise_xor(image):
    # random images color numpy array
    img = np.random.randint(0, 255, (image.shape[0], image.shape[1], 3), dtype=np.uint8)    
    bit_xor = cv2.bitwise_xor(image, img)
    return bit_xor, img