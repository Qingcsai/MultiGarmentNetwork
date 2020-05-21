# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle as pkl
import numpy as np
import os
from os.path import join, split
from PIL import Image
import sys

NUM = 8
# 数据编号
CQ_DATA_FLAG = 18

# 从snapshot得到分割图并保存
def step_1_create_segment():
    print('step_1_create_segment')
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
        
# 生成720*720大小分割图并保存
def step_2_images_crop(path0, savedpath):
    print('step_2_images_crop')
    path = path0
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
def step_3_create_J_2d(input_pkl):
    print('step_3_create_J_2d')
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

def change_render_num(input_pkl,num):
    input_pkl['rendered'] = input_pkl['rendered'][:,:,:,:,:num]
    
def step_5_create_rendered_as_pkl(input_pkl):
    print('step_5_create_rendered_as_pkl')
    for i in range(NUM):
        input_pkl['rendered'][:,:,:,:,i] = input_pkl['image_{}'.format(i)].astype('float32')
    return input_pkl

def main(argv):
    #inp = pkl.load(open('../assets/test_data.pkl', 'rb'), encoding='latin1')
    '''
    images = [inp['image_{}'.format(i)].astype('float32') for i in range(NUM)]
    J_2d = [inp['J_2d_{}'.format(i)].astype('float32') for i in range(NUM)]
    vertex_label = inp['vertexlabel'].astype('int64')
    '''
    cq_inp_test = pkl.load(open('./cq_inp_2_new.pkl', 'rb'), encoding='latin1')
    cq_inp = cq_inp_test

    ############################
    if argv[1] == 'step1to2':
        print('step1to2')
        
        step_1_create_segment()
        # 原快照图做crop
        step_2_images_crop('data_{}/snapshot/'.format(CQ_DATA_FLAG), 'data_{}/snapshot_crop/'.format(CQ_DATA_FLAG))
        # 分割图做crop
        step_2_images_crop('data_{}/snapshot_segment/'.format(CQ_DATA_FLAG), 'data_{}/snapshot_segment_crop/'.format(CQ_DATA_FLAG))


    if argv[1] == 'step3to5':
        print('step3to5')
        
        # 将J_2d保存到pkl文件中
        cq_inp = step_3_create_J_2d(cq_inp)
        # 保存image_i到pkl文件
        cq_inp = step_4_images_crop_save_as_pkl(cq_inp)
        # 保存rendered到pkl文件
        cq_inp = step_5_create_rendered_as_pkl(cq_inp)
        pkl.dump(cq_inp, open('cq_inp_{}.pkl'.format(CQ_DATA_FLAG),'wb'), pkl.HIGHEST_PROTOCOL)
        
        
    if argv[1] == 'valid':
        print('valid')
        
        #检查刚生成的pkl是否正确
        cq_eval = pkl.load(open('./cq_inp_{}.pkl'.format(CQ_DATA_FLAG), 'rb'), encoding='latin1')
        #检查image_i是否生成正确
        cq_images = [cq_eval['image_{}'.format(i)].astype('float32') for i in range(NUM)]
        save_pkl_as_pic(cq_images)
        #检查rendered是否生成正确
        #save_pkl_as_pic(cq_eval['rendered'])
        

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)
