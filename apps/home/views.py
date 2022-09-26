# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse

from apps.home.models import Image as ModelImage
from apps.home.forms import ImageCreate

from pathlib import Path
from statistics import mean
from skimage import io

from .ImageOperation import *

import cv2
import numpy as np
import matplotlib.image as mimg


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
        context['data'] = getData(load_template, request)        

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    # except:
    #     html_template = loader.get_template('home/page-500.html')
    #     return HttpResponse(html_template.render(context, request))

# Core data for each template
def getData(file_template, request):
    data = []

    # Return different data for each template names
    if file_template == 'pertemuan-2.html':
        data = getInfoImg(range(0, 3))
    elif file_template == 'pertemuan-3.html':
        data = getCropAndInvertImage()
    elif file_template == 'pertemuan-4.html':
        # Get image file from requesting
        if request.method == 'POST':
            data = {}
            if request.FILES['image']:
                upload = ImageCreate(request.POST, request.FILES)
                fileImg = request.FILES['image']                
                image = ModelImage.objects.create(image=fileImg, task="Pertemuan-4")
                image.save()                
                data['is_upload'] = True
                # Split url to get image name
                url = str(image.image).split('/')
                data['image'] = url[5]              
                data['rgb'] = getAllOperationFrom(fileImg, request)  
                data['addition'] = request.POST['addition']
                data['substraction'] = request.POST['substraction']
                data['multiplication'] = request.POST['multiplication']
                data['division'] = request.POST['division']
        else:
            data = {}
            data['rgb'] = getAllOperationFromDefault("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png")
            data['is_upload'] = False
    
    return data

def getAllOperationFromDefault(url):    
    save_path_to = str(Path(__file__).resolve().parent.parent) + \
        '\\static\\assets\\images\\brightness\\'
    data = {}
    
    rgb_img = io.imread(url) # read image using skimage
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) # convert to bgr color    

    data['original'] = rgb_img
    
    # Operation add
    add_img = rgb_add_np(rgb_img)
    add_img_cv = rgb_add_cv(bgr_img)
    data['add_np'] = add_img
    data['add_cv'] = add_img_cv

    # Save image
    Image.fromarray(add_img).save(save_path_to+"add_np.jpg")    
    cv2.imwrite(save_path_to+'add_cv.jpg', add_img_cv)    

    # Operation substract  
    sub_img = rgb_substraction_np(rgb_img)
    sub_img_cv = rgb_substraction_cv(bgr_img)
    data['sub_np'] = sub_img
    data['sub_cv'] = sub_img_cv

    # Save image
    Image.fromarray(sub_img).save(save_path_to+"sub_np.jpg")    
    cv2.imwrite(save_path_to+'sub_cv.jpg', sub_img_cv)

    # Operation multiplication
    mul_img = rgb_multiplication_np(rgb_img)
    mul_img_cv = rgb_multiplication_cv(bgr_img)
    data['mul_np'] = mul_img
    data['mul_cv'] = mul_img_cv

    # Save image
    Image.fromarray(mul_img).save(save_path_to+"mul_np.jpg")    
    cv2.imwrite(save_path_to+'mul_cv.jpg', mul_img_cv)    

    # Operation division
    div_img = rgb_division_np(rgb_img)
    div_img_cv = rgb_division_cv(bgr_img)
    data['div_np'] = div_img
    data['div_cv'] = div_img_cv

    # Save image
    Image.fromarray(div_img).save(save_path_to+"div_np.jpg")    
    cv2.imwrite(save_path_to+'div_cv.jpg', div_img_cv)    

    # Operation bitwise and
    and_img = bitwise_and(bgr_img)

    # Save image
    cv2.imwrite(save_path_to+'and.jpg', and_img[0])
    data['and'] = and_img[0]
    data['and_rand'] = and_img[1]


    # Operation bitwise or
    or_img = bitwise_or(bgr_img)

    # Save image
    cv2.imwrite(save_path_to+'or.jpg', or_img[0])
    data['or'] = or_img[0]
    data['or_rand'] = or_img[1]

    # Operation bitwise xor
    xor_img = bitwise_xor(bgr_img)

    # Save image
    cv2.imwrite(save_path_to+'xor.jpg', xor_img[0])
    data['xor'] = xor_img[0]
    data['xor_rand'] = xor_img[1]

    # Operation bitwise not
    not_img = bitwise_not(bgr_img)

    # Save image
    cv2.imwrite(save_path_to+'not.jpg', not_img[0])
    data['not'] = not_img[0]
    data['not_rand'] = not_img[1]
    
    return data

def getAllOperationFrom(url, request):    
    save_path_to = str(Path(__file__).resolve().parent.parent) + \
        '\\static\\assets\\images\\brightness\\'
    data  = {}

    rgb_img = io.imread(url) # read image using skimage
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) # convert to bgr color    

    data['original'] = rgb_img
   
    # Operation add
    add_img = rgb_add_np(rgb_img, int(request.POST['addition']))
    add_img_cv = rgb_add_cv(bgr_img, int(request.POST['addition']))
    data['add_np'] = add_img
    data['add_cv'] = add_img_cv

    # Save image
    Image.fromarray(add_img).save(save_path_to + 'add_np.jpg')
    cv2.imwrite(save_path_to+'add_cv.jpg', add_img_cv)    

    # Operation substract  
    sub_img = rgb_substraction_np(rgb_img, int(request.POST['substraction']))
    sub_img_cv = rgb_substraction_cv(bgr_img, int(request.POST['substraction']))
    data['sub_np'] = sub_img
    data['sub_cv'] = sub_img_cv    

    # Save image
    Image.fromarray(sub_img).save(save_path_to + 'sub_np.jpg')    
    cv2.imwrite(save_path_to+'sub_cv.jpg', sub_img_cv)    

    # Operation multiplication
    mul_img = rgb_multiplication_np(rgb_img, int(request.POST['multiplication']))
    mul_img_cv = rgb_multiplication_cv(bgr_img, int(request.POST['multiplication']))
    data['mul_np'] = mul_img
    data['mul_cv'] = mul_img_cv    

    # Save image
    Image.fromarray(mul_img).save(save_path_to + 'mul_np.jpg')
    cv2.imwrite(save_path_to+'mul_cv.jpg', mul_img_cv)    

    # Operation division
    div_img = rgb_division_np(rgb_img, int(request.POST['division']))
    div_img_cv = rgb_division_cv(bgr_img, int(request.POST['division']))
    data['div_np'] = div_img
    data['div_cv'] = div_img_cv

    # Save image
    Image.fromarray(div_img).save(save_path_to + 'div_np.jpg')
    cv2.imwrite(save_path_to+'div_cv.jpg', div_img_cv)    

    # Operation bitwise and
    and_img = bitwise_and(bgr_img)

    # Save image
    cv2.imwrite(save_path_to+'and.jpg', and_img[0])
    data['and'] = and_img[0]
    data['and_rand'] = and_img[1]

    # Operation bitwise or
    or_img = bitwise_or(bgr_img)

    # Save image
    cv2.imwrite(save_path_to+'or.jpg', or_img[0])
    data['or'] = or_img[0]
    data['or_rand'] = or_img[1]

    # Operation bitwise xor
    xor_img = bitwise_xor(bgr_img)

    # Save image
    cv2.imwrite(save_path_to+'xor.jpg', xor_img[0])
    data['xor'] = xor_img[0]
    data['xor_rand'] = xor_img[1]

    # Operation bitwise not
    not_img = bitwise_not(bgr_img)

    # Save image
    cv2.imwrite(save_path_to+'not.jpg', not_img[0])
    data['not'] = not_img[0]
    data['not_rand'] = not_img[1]
    
    return data

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

    for i in range(w):
        for j in range(h):
            # ratio of RGB will be between 0 and 1
            lst = [float(m[i][j][0]), float(m[i][j][1]), float(m[i][j][2])]
            avg = float(mean(lst))
            newImage[i][j][0] = avg
            newImage[i][j][1] = avg
            newImage[i][j][2] = avg
            # newImage[i][j][3] = 1 # alpha value to be 1
    
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