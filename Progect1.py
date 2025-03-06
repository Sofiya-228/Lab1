import numpy as np
from PIL import Image, ImageOps
import math


#1 task
image = np.zeros((200,200,3), dtype = np.uint8)
img = Image.fromarray(image,mode = 'RGB')
img.save('img1.png')

image = np.full((200,200,3),255, dtype= np.uint8)
img = Image.fromarray(image,mode = 'RGB')
img.save('img2.png')

image = np.full((200,200,3), (255,0,0), dtype=np.uint8)
img = Image.fromarray(image,mode = 'RGB')
img.save('img3.png')

#2 task
#4
def dotted_line(self, image, x0, y0, x1, y1, count, color):
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color
#5
def dotted_line_v2(self, image, x0, y0, x1, y1, color):
    count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color
#6
def x_loop_line(self, image, x0, y0, x1, y1, color):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color
#7
def x_loop_line_hotfix_1(self, image, x0, y0, x1, y1, color):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color
#8
def x_loop_line_hotfix_2(self, image, x0, y0, x1, y1, color):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    xchange = False

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)

        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

#9
def x_loop_line_v2(self, image, x0, y0, x1, y1, color):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)

        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

#10
def x_loop_line_v2_no_y_calc(self, image, x0, y0, x1, y1, color):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        xchange = False
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            xchange = True

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        y = y0
        dy = abs(y1 - y0) / (x1 - x0)
        derror = 0.0
        y_update = 1 if y1 > y0 else -1

        for x in range(x0, x1):
            if xchange:
                image[x, y] = color
            else:
                image[y, x] = color

            derror += dy
            if derror > 0.5:
                derror -= 1.0
                y += y_update

#11
def x_loop_line_v2_no_y_calc_for_some_unknown_reason(self, image, x0, y0, x1, y1, color):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        xchange = False
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            xchange = True

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        y = y0

        dy = 2.0 * (x1 - x0) * abs(y1 - y0) / (x1 - x0)
        derror = 0.0
        y_update = 1 if y1 > y0 else -1

        for x in range(x0, x1):
            if xchange:
                image[x, y] = color
            else:
                image[y, x] = color

            derror += dy
            if derror > 2.0 * (x1 - x0) * 0.5:
                derror -= 2.0 * (x1 - x0) * 1.0
                y += y_update

#12
def bresenham_line(self, image, x0, y0, x1, y1, color):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0

    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if derror > (x1 - x0):
            derror -= 2 * (x1 - x0)
            y += y_update


img_mat = np.zeros((200, 200, 3), dtype = np.uint8)
for i in range(13):
    x0 = 100
    y0 = 100
    x1 = 100 + 95*math.cos((i*2*math.pi)/13)
    y1 = 100 + 95*math.sin((i*2*math.pi)/13)
    bresenham_line(None,img_mat, x0,y0,x1,y1, 255)

img = Image.fromarray(img_mat, mode = 'RGB')
img.save('img12.2.png')


#3 task
val = []
with open("model_1.obj", "r") as f:
    for line in f:
        if line.startswith('v '):
            parts = line.split()   #разделение по пробелам
            val.append([float(parts[1]), float(parts[2]), float(parts[3])])

#4 task
img_mat_13 = np.zeros((2000, 2000, 3), dtype = np.uint8)
for i in val:
    x= 5000*i[0]+1000
    y= 5000*i[1]+1000
    img_mat_13[int(y), int(x)]=255
img_13 = Image.fromarray(img_mat_13, mode = 'RGB')
img_13 = ImageOps.flip(img_13)
img_13.save('img13.png')


#5 task
val_2 = []
with open("model_1.obj", "r") as f:
    for line in f:
        if line.startswith('f '):
            elem = line.split()
            val_2.append([int(elem[1].split('/')[0]), int(elem[2].split('/')[0]), int(elem[3].split('/')[0])])


#6 task
img_mat_14 = np.zeros((2000, 2000, 3), dtype = np.uint8)
for f in range(len(val_2)):
    x1 = int(val[(val_2[f][0]-1)][0] * 10000 + 1000)
    y1 = int(val[(val_2[f][0]-1)][1] * 10000 + 1000)
    x2 = int(val[(val_2[f][1]-1)][0] * 10000 + 1000)
    y2 = int(val[(val_2[f][1]-1)][1] * 10000 + 1000)
    x3 = int(val[(val_2[f][2]-1)][0] * 10000 + 1000)
    y3 = int(val[(val_2[f][2]-1)][1] * 10000 + 1000)

    bresenham_line(None,img_mat_14, x1, y1, x2, y2, 255)
    bresenham_line(None, img_mat_14, x1, y1, x3, y3, 255)
    bresenham_line(None, img_mat_14, x2, y2, x3, y3, 255)

img_14 = Image.fromarray(img_mat_14, mode='RGB')
img_14 = ImageOps.flip(img_14)
img_14.save('img14.png')
