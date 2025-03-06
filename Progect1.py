import numpy as np
from PIL import Image
import math

image = np.zeros((200,200,3), dtype = np.uint8)
img = Image.fromarray(image,mode = 'RGB')
img.save('img1.png')

image = np.full((200,200,3),255, dtype= np.uint8)
img = Image.fromarray(image,mode = 'RGB')
img.save('img2.png')

image = np.full((200,200,3), (255,0,0), dtype=np.uint8)
img = Image.fromarray(image,mode = 'RGB')
img.save('img3.png')

def dotted_line(self, image, x0, y0, x1, y1, count, color):
    step = 1.0/count
    for t in np.arange (0,1, step):
        x = round ((1.0-t)*x0 + t*x1)
        y = round ((1.0-t)*y0+t*y1)
        image[y,x] = color


def dotted_line(self, image, x0, y0, x1, y1,
                color):
    count = math.sqrt((x0 - x1) ** 2 + (y0-y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
    y = round((1.0 - t) * y0 + t * y1)
    image[y, x] = color

def x_loop_line(self, image, x0, y0, x1, y1,
color):
 for x in range (x0, x1):
 t = (x-x0)/(x1 - x0)
 y = round ((1.0 - t)*y0 + t*y1)
 image[y, x] = color
 