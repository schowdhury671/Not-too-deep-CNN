
import os
#import matplotlib.pyplot as plt
#import PIL
from PIL import Image
#import numpy as np

max_row_pix = 0
max_col_pix = 0
min_row_pix = 500000
min_col_pix = 500000
aspect_ratio = 1.5
row = 32
col = 32


def dimensions(image_name):
    i = Image.open(image_name)
    imgSize = i.size 
    print(imgSize)
    
    pix = np.array(i)
    print(pix.shape[0],pix.shape[1],pix.shape[2])
    
    global max_row_pix
    global max_col_pix
    global min_row_pix
    global min_col_pix
       
    i = Image.open(image_name)
    imgSize = i.size 
    
    #print(imgSize)
    if(imgSize[0] > max_row_pix):
        max_row_pix = imgSize[0]
    if(imgSize[0] > max_col_pix):    
        max_col_pix = imgSize[1]
    if(imgSize[0] < min_row_pix):
        min_row_pix = imgSize[0]
    if(imgSize[0] < min_col_pix):    
        min_col_pix = imgSize[1]
    
    #basewidth = 300
    #img = Image.open('somepic.jpg')
    #wpercent = (basewidth/float(img.size[0]))
    #hsize = int((float(img.size[1])*float(wpercent)))
    resize(image_name)
    
    
def resize(image_name):    
    
    global row
    global col

    #col = int(row / aspect_ratio)
    i = Image.open(image_name)
    i = i.resize((row,col), PIL.Image.ANTIALIAS)
    i.save(image_name)

    write_pixel(image_name)


def write_pixel(image_name):

    im=Image.open(image_name)
    fil = open('reshaped_image_pixel.csv', 'w')
    pixel = im.load()
    
    print("image name is ",image_name)
    #print(os.path.splitext(image_name)[-2])
    #img = image_name.split('/')
    #print(img[0])
    #print(img[1])
    
    row,column = im.size
    #fil.write('[')
    for x in range(row):
        for y in range(col):
            pix=str(pixel[x,y])
            #print(pix)
            pix = pix.replace(' ','')
            pix = pix.replace('(','')
            pix = pix.replace(')','')
            fil.write(pix) 
            #if(y != col -1):                  
            fil.write(',')
            #ct += 1
        #fil.write('\n\n*********one line completes here!!********\n\n')    
    #fil.write('1')
    fil.write('\n')        
    fil.close()
    
def main():
        
    ct = 0
    for root, dirs, files in os.walk("4_convnets/32_net_test/faces", topdown=True):
        for name in files:
            write_pixel(os.path.join(root, name))
            ct += 1

    
    print("**************Reshaped image pixel writing*************")        
    write_pixel("temp_image.jpg")
    print("**************Reshaped image pixel written*************")
    #print('printed elements should be 1113279, found ',ct)
    #print("\nFinal max row is ",max_row_pix,"\nFinal max col is ",max_col_pix)        
    #print("\nFinal min row is ",min_row_pix,"\nFinal min col is ",min_col_pix)        
            
main()
