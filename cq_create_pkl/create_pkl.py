# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle as pkl
import numpy as np
import os
from os.path import join, split

inp = pkl.load(open('../assets/test_data.pkl', 'rb'), encoding='latin1')
#cq_inp = pkl.load(open('./cq_inp.pkl', 'rb'), encoding='latin1')
cq_inp_test = pkl.load(open('./cq_inp_2_new.pkl', 'rb'), encoding='latin1')
cq_inp = cq_inp_test
NUM = 8
CQ_DATA_FLAG = 18

images = [inp['image_{}'.format(i)].astype('float32') for i in range(NUM)]
J_2d = [inp['J_2d_{}'.format(i)].astype('float32') for i in range(NUM)]
vertex_label = inp['vertexlabel'].astype('int64')


'''
images2 = images[0]
images1 = images[0][1] # images[frames][batch_th]
J_2d_0 = J_2d[0][0] # J_2d[frames][batch_th]
'''

from PIL import Image


#pkl.dump(inp2, open('cq_inp2.pkl','wb'), pkl.HIGHEST_PROTOCOL)
def read_J_2d(txt_path):
    #txt_path = ".\snapshot\snapshot00.txt"
    f = open(txt_path, 'r')
    lineInfo = f.readlines()
    dataset = []
    for line in lineInfo:
        temp1 = line.strip('\n') #去除指定字符
        temp2 = temp1.split() #指定分隔符，默认所有空字符
        dataset += temp2
    f.close()
    data_output = list(map(float,dataset)) #python 3.x
    
    return data_output

#J_2d存到pkl文件
def step1_create_J_2d(input_pkl):
    print('step1_create_J_2d')
    path = 'data_{}/snapshot_j2d/'.format(CQ_DATA_FLAG)
    filelist = os.listdir(path)
    flag = 0

    for item in filelist:
        txt_path = path + item
        print(txt_path)
        J_2d_a = np.array(read_J_2d(txt_path))
        #print(J_2d_a)
        
        # 归一化
        J_2d_b = J_2d_a.reshape(-1,3)
        resolution=[720, 720]
        J_2d_b[:, 2] /= np.expand_dims(np.mean(J_2d_b[:, 2][J_2d_b[:, 2] > 0.1]), -1)
        J_2d_b = J_2d_b * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
        J_2d_b[:, 0] *= 1. * resolution[1] / resolution[0]
        print(J_2d_b)
        
        J_2d_name = 'J_2d_' + str(flag)
        input_pkl[J_2d_name][0] = J_2d_b
        flag += 1
    return input_pkl
        
#cq_inp = step1_create_J_2d(cq_inp)

# 保存image_n为RGB图
#cq_images = [cq_inp_test['image_{}'.format(i)].astype('float32') for i in range(NUM)]
#cq_images = [cq_inp['image_{}'.format(i)].astype('float32') for i in range(NUM)]
def save_img(A,img_name = 'img_name'):
    B = A * 255
    im = Image.fromarray(np.uint8(B),mode='RGB')
    im.save(img_name)
    
def save_pkl_as_pic(images):
    Batch_size = 1
    savedpath = 'data_{}/evaluate_seg/'.format(CQ_DATA_FLAG)
    for i in range(Batch_size):
        for n in range(NUM):
            img_name = savedpath + 'cq_seg_img'+ str(i) + str(n) + '.jpeg'
            print(img_name)
            #save_img(images[n][i], img_name)
            save_img(images[i,:,:,:,n], img_name)

save_pkl_as_pic(cq_inp['rendered'])

# 从snapshot得到分割图并保存
def step_2_create_segment():
    print('step_2_create_segment')
    path = 'data_{}/snapshot/'.format(CQ_DATA_FLAG)
    savedpath = 'data_{}/snapshot_segment/'.format(CQ_DATA_FLAG)
    filelist = os.listdir(path)
    for item in filelist:
        im = Image.open( path + item )#打开图片
        width = im.size[0]#获取宽度
        height = im.size[1]#获取长度
        for x in range(width):
            for y in range(height):
                r,g,b = im.getpixel((x,y))	
                # Pants (65, 0, 65), Short-Pants (0, 65, 65), Shirt (145, 65, 0), T-Shirt (145, 0, 65) and Coat (0, 145, 65).
                if r>0 and g==0 and b==0: #shoes
                    im.putpixel((x,y),(255,255,255)) #skin
                if r>=50 and g>0 and b==0:  #upper-cloth
                    #im.putpixel((x,y),(145, 0, 65)) #t-shirt
                    im.putpixel((x,y),(145, 65, 0)) #Shirt
                    #im.putpixel((x,y),(0, 145, 65)) #Coat
                if r==0 and g==0 and b>0: #skin
                    im.putpixel((x,y),(255,255,255)) #skin
                if r==0 and (g>80 or b>80): #hair
                    im.putpixel((x,y),(255,255,255)) #skin
                if r==0 and g<80 and b<80 and g>0 and b>0: #down-cloth
                    im.putpixel((x,y),(65, 0, 65)) #pants
                    #im.putpixel((x,y),(0, 65, 65))# Short-Pants (0, 65, 65)
                    
        im = im.convert('RGB')
        im.save(savedpath + item)
        print('item of %s is saved '%(item))

#step_2_create_segment()
    
# 生成720*720大小分割图并保存
def step_3_images_crop(path0, savedpath0):
    print('step_3_images_crop')
    path = path0
    savedpath = savedpath0
    '''
    #存snapshot_crop给J_2d_x用
    path = 'data_{}/snapshot/'.format(CQ_DATA_FLAG)
    savedpath = 'data_{}/snapshot_crop/'.format(CQ_DATA_FLAG)
    
    #给imgae_x用
    path = 'data_{}/snapshot_segment/'.format(CQ_DATA_FLAG)
    savedpath = 'data_{}/snapshot_segment_crop/'.format(CQ_DATA_FLAG)
    '''
    filelist = os.listdir(path)
    for item in filelist:
        img = Image.open( path + item )#打开图片
        half_the_width = img.size[0] // 2
        half_the_height = img.size[1] // 2
        img_crop = img.crop(
            (
                half_the_width - 460,
                half_the_height - 460,
                half_the_width + 460,
                half_the_height + 460
            )
        )
        img_crop = img_crop.resize((720,720),Image.ANTIALIAS)
        print("默认缩放NEARESET",img_crop.size)
        img_crop.save(savedpath + item)

#step_3_images_crop()

# 分割图存到pkl文件
def step_4_images_crop_save_as_pkl(input_pkl):
    print('step_4_images_crop_save_as_pkl')
    path = 'data_{}/snapshot_segment_crop/'.format(CQ_DATA_FLAG)
    filelist = os.listdir(path)
    flag = 0
    for item in filelist:
        print(path + item)
        img = Image.open( path + item )#打开图片
        img_array = np.asarray(img, dtype='float32') / 255
        print(img_array.shape)
        image_name = 'image_' + str(flag)
        #input_pkl[image_name][0] = img_array
        input_pkl[image_name][0] = img_array
        flag += 1
    return input_pkl
        
#cq_inp = step_4_images_crop_save_as_pkl(cq_inp)
    

def change_render_num(input_pkl,num):
    input_pkl['rendered'] = input_pkl['rendered'][:,:,:,:,:num]
    
def step_5_create_rendered_as_pkl(input_pkl):
    print('step_5_create_rendered_as_pkl')
    for i in range(NUM):
        input_pkl['rendered'][:,:,:,:,i] = input_pkl['image_{}'.format(i)].astype('float32')
    return input_pkl

#cq_inp = step_5_create_rendered_as_pkl(cq_inp)
#pkl.dump(cq_inp, open('cq_inp_4.pkl','wb'), pkl.HIGHEST_PROTOCOL)

#cq_inp_4 = pkl.load(open('./cq_inp_4.pkl', 'rb'), encoding='latin1')

#cq_inp_new = {k: v[:1] for k, v in cq_inp_new.items()} # added this to slice
#pkl.dump(cq_inp_new, open('cq_inp_new_1.pkl','wb'), pkl.HIGHEST_PROTOCOL)
    
#cq_inp_1 = pkl.load(open('./cq_inp_test.pkl', 'rb'), encoding='latin1')

#cq_inp_1_new = {k: v[:1] for k, v in cq_inp_1.items()} # added this to slice
#pkl.dump(cq_inp_1_new, open('cq_inp_1_new.pkl','wb'), pkl.HIGHEST_PROTOCOL)
#cq_inp_1 = pkl.load(open('./cq_inp_1.pkl', 'rb'), encoding='latin1')

############################
step_2_create_segment()
step_3_images_crop('data_{}/snapshot/'.format(CQ_DATA_FLAG), 'data_{}/snapshot_crop/'.format(CQ_DATA_FLAG))
step_3_images_crop('data_{}/snapshot_segment/'.format(CQ_DATA_FLAG), 'data_{}/snapshot_segment_crop/'.format(CQ_DATA_FLAG))

'''
######################

cq_inp = step1_create_J_2d(cq_inp)
cq_inp = step_4_images_crop_save_as_pkl(cq_inp)
cq_inp = step_5_create_rendered_as_pkl(cq_inp)
pkl.dump(cq_inp, open('cq_inp_{}.pkl'.format(CQ_DATA_FLAG),'wb'), pkl.HIGHEST_PROTOCOL)

cq_images = [cq_inp['image_{}'.format(i)].astype('float32') for i in range(NUM)]
save_pkl_as_pic(cq_images)
cq_eval = pkl.load(open('./cq_inp_{}.pkl'.format(CQ_DATA_FLAG), 'rb'), encoding='latin1')
'''