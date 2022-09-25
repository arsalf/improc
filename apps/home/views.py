# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse

from asyncio.windows_events import NULL
from django.shortcuts import render
from django.http import HttpResponse
import cv2
import numpy as np
from pathlib import Path
from statistics import mean
import matplotlib.image as mimg
import pandas as pd
from skimage import io
from PIL import Image 
import matplotlib.pylab as plt

@login_required(login_url="/login/")
def index(request):
    context = {'segment': 'index'}

    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        # Load data for each template
        context['data'] = getData(load_template)

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))

# Core data for each template
def getData(file_template):
    data = []

    # Return different data for each template names
    if file_template == 'pertemuan-2.html':
        data = getInfoImg(range(0, 3))
    elif file_template == 'pertemuan-3.html':
        data = getCropAndInvertImage()
    elif file_template == 'pertemuan-4.html':
        data = getImgFromUrl("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png")    
    
    return data

def getImgFromUrl(url):    
    save_path_to = str(Path(__file__).resolve().parent.parent) + \
        '\\static\\assets\\images\\merge\\'
    image_0 = io.imread(url) # read image use skimage
    image_2 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB) #convert bgr to rgb
    final_frame = cv2.hconcat((image_0, image_2)) # merge image
    status = cv2.imwrite(save_path_to+"lena.jpg", final_frame)

    # add brightness_add
    image_1 = brightness_add(image_0)
    save_path_to = str(Path(__file__).resolve().parent.parent) + \
        '\\static\\assets\\images\\brightness\\'
    image_1.save(save_path_to+"lena.jpg")    

    # add brightness_addcv    
    image_3 = brightness_addcv(image_2)
    status = cv2.imwrite(save_path_to+"lenacv.jpg", image_3)

    # add brightness_addcv
    image_4 = brightness_add(image_2)    
    image_4.save(save_path_to+"lenacvmanual.jpg")    

    # add brightness_subtraction
    image_5 = brightness_subtraction(image_2)
    image_5 = Image.fromarray(image_5)
    image_5.save(save_path_to+"lenasub.jpg")

    # add brightness_subtractioncv
    image_6 = brightness_subtractioncv(image_0)
    status = cv2.imwrite(save_path_to+"lenasubcv.jpg", image_6)    

    # add brightness_multiplication
    image_7 = brightness_multiplication(image_0)
    image_7.save(save_path_to+"lenamul.jpg")

    return status

def brightness_subtraction(image):
    image = image.astype('uint16')
    image = image-100
    image = np.clip(image, 0, 255)
    image = image.astype('uint8')
    return image

def brightness_subtractioncv(image):
    new_image = cv2.subtract(image, 100)
    new_image = np.clip(new_image, 0, 255)
    return new_image

def brightness_multiplication(image):
    image = np.asarray(image).astype('uint16')
    image = image*1.25
    image = np.clip(image, 0, 255)
    new_image = image.astype('uint8')
    new_image = Image.fromarray(new_image)
    return new_image

def brightness_multiplicationcv(image):
    new_image = cv2.multiply(image, 1.25)
    new_image= np.clip(new_image, 0, 255)
    return new_image

def brightness_dividecv(image):
    new_image = cv2.divide(image, 2)
    new_image= np.clip(new_image, 0, 255)
    return new_image

def brightness_divide(image):
    image = np.asarray(image).astype('uint16')
    image = image/2
    image = np.clip(image, 0, 255)
    new_image = image.astype('uint8')
    #new_image = Image.fromarray(new_image)
    return new_image

def bitwise_and(image):
    bit_and = cv2.bitwise_and(image, image)
    return bit_and

def bitwise_or(image):
    bit_or = cv2.bitwise_or(image, image)
    return bit_or

def bitwise_not(image):
    bit_not = cv2.bitwise_not(image)
    return bit_not

def bitwise_xor(image):
    bit_xor = cv2.bitwise_xor(image, image)
    return bit_xor

def brightness_add(image):
    image = np.asarray(image).astype('uint16')
    image = image+100
    image = np.clip(image, 0, 255)
    new_image = image.astype('uint8')
    new_image = Image.fromarray(new_image)
    return new_image

def brightness_addcv(image):
    new_image = cv2.add(image, 100)
    return new_image

def getCropAndInvertImage():
    status = getCropImage()
    if status:
        status = getInvertImage()

    return status

def getInvertImage():
    path = str(Path(__file__).resolve().parent.parent)+'\\static\\assets\\images\\'
    save_path_to = str(Path(__file__).resolve().parent.parent) + \
        '\\static\\assets\\images\\invert\\'

    img = cv2.imread(path+"MyPhoto.jpg", cv2.IMREAD_COLOR)
    img_invert = cv2.bitwise_not(img)
    status = cv2.imwrite(save_path_to+"MyPhoto.jpg", img_invert)

    return status

def getCropImage():
    path = str(Path(__file__).resolve().parent.parent)+'\\static\\assets\\images\\'
    save_path_to = str(Path(__file__).resolve().parent.parent) + \
        '\\static\\assets\\images\\crop\\'

    img = cv2.imread(path+"MyPhoto.jpg", cv2.IMREAD_COLOR)
    img_crop = img[80:, 560:1200] # Cukup di slice aja    
    status = cv2.imwrite(save_path_to+"MyPhoto.jpg", img_crop)

    return status

def getInfoImg(rData):
    data = []
    for r in rData:
        res = processing(str(r)+".jpg", 5)
        i = 1
        strFormat = ""
        for items in res[0]:
            for colors in items:
                strFormat += "<tr><th>"+str(i)+"</th><td>"+str(colors[2])+"</td><td>"+str(
                    colors[1])+"</td><td>"+str(colors[0])+"</td></tr>"
                i += 1
        imgGray = getGrayImgOpenCv(str(r)+".jpg")
        # imgGrayManual = getGrayImgManual(str(r)+".jpg")

        data.append({
            "nama_file": str(r)+".jpg",
            "res": strFormat,
            "width": res[0].shape[1],
            "height": res[0].shape[0],
            "rwidth": res[1].shape[1],
            "rheight": res[1].shape[0],
            "gray_opencv": imgGray,
        })
    getDetectedObject()
    return data


def processing(file_name, scale_percent):
    path = str(Path(__file__).resolve().parent.parent)+'\\static\\images\\'+file_name

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    results = resizeImg(img, scale_percent)

    return results, img


def resizeImg(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized


def getGrayImgOpenCv(file_name):
    path = str(Path(__file__).resolve().parent.parent)+'\\static\\images\\'+file_name
    save_path_to = str(Path(__file__).resolve().parent.parent) + \
        '\\static\\images\\gray-opencv\\'+file_name

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # save the image
    status = cv2.imwrite(save_path_to, gray)

    return status


def getGrayImgManual(file_name):
    path = str(Path(__file__).resolve().parent.parent)+'\\static\\images\\'+file_name
    save_path_to = str(Path(__file__).resolve().parent.parent) + \
        '\\static\\images\\gray-manual\\'+file_name

    m = mimg.imread(path)

    # determining width and height of original image
    w, h = m.shape[:2]

    # new Image dimension with 4 attribute in each pixel
    newImage = np.zeros([w, h, 3])
    print(w)
    print(h)

    for i in range(w):
        for j in range(h):
            # ratio of RGB will be between 0 and 1
            lst = [float(m[i][j][0]), float(m[i][j][1]), float(m[i][j][2])]
            avg = float(mean(lst))
            newImage[i][j][0] = avg
            newImage[i][j][1] = avg
            newImage[i][j][2] = avg
            # newImage[i][j][3] = 1 # alpha value to be 1
    print(newImage)
    # Save image using imsave
    status = cv2.imwrite(save_path_to, newImage)

    return status


def getDetectedObject():
    path = str(Path(__file__).resolve().parent.parent) + \
        '\\static\\images\\tantangan.jpg'
    # save_path_to = str(Path(__file__).resolve().parent.parent)+'\\static\\images\\gray-opencv\\'+file_name

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bgBGR = np.array(245)
    row = img.shape[0]
    col = img.shape[1]
    data_horizon = []
    data_vertical = []

    for i in range(0, row):
        for j in range(0, col):            
            # print(img[i][j])
            if(np.greater_equal(img[i][j], bgBGR) == False):
                data_horizon.append({
                    "x": i,
                    "y": j,
                    "rgb": img[i][j]
                })

    print(data_horizon[0])

    for i in range(0, col):
        for j in range(0, row):                        
            if(np.greater_equal(img[j][i], bgBGR) == False):    
                data_vertical.append({
                    "x": j,
                    "y": i,
                    "rgb": img[j][i]
                })                              
    print(data_vertical[0])