#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:51:01 2022

@author: leovain
"""

##Import packages
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir, makedirs, path
from os.path import isfile, join, dirname, exists, splitext
from pathlib import Path
from matplotlib.pyplot import imsave
from sklearn import preprocessing as prep
import cv2
import imutils
import cmapy

global APPENDIX
APPENDIX = ['v_signal', 'h_signal', 'ROI location', 'y or height', 'x or width', 'center or top', 'periderm or down', 'offset']

def get_filelist(filepath, img_type = '.jpg'):
    #Get files of chosen directory
    files_in_dir = [ i for i in listdir(filepath) if isfile(join(filepath, i)) ]
    #for loop for getting img_type images from filelist
    img_files = []
    for i in range (0,len(files_in_dir)):
        chk = files_in_dir[i]
        if chk.find(img_type) != -1:
            img_files.append(files_in_dir[i])
    img_files.sort()        

    return img_files

def get_e_distance(x1, y1, x2, y2):
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return d

def get_contour_areas(contours):
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    return areas

def pad_image_from_centerpoint(img, channel = 1):
    #creates an image with equal distance to x and y edges from predetermined center point
    #orig = img.copy()
    img.setflags(write = True)
    img_data = img[:,:,channel].astype('uint8')
    dim = img_data.shape
    #threshold the positional information
    ret, thresh = cv2.threshold(img_data, 254, 255, 0)
    #get contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #get countour ares
    areas = get_contour_areas(contours)
    #get smallest area contour position
    min_pos = np.argmin(areas)
    #waringin if dot is lare
    if areas[min_pos] > 20:
        print('Smallest marker area too large, assuming middle pixel as middlepoint')
        x = int(int(dim[1]/2))
        y = int(int(dim[0]/2))
    else:
        M = cv2.moments(contours[min_pos])
        #get the centerpoint of the smallest area
        if M["m00"] == 0:
            print('Smallest marker area too small or not found, assuming middle pixel as middlepoint')
            x = int(int(dim[1]/2))
            y = int(int(dim[0]/2))
        else:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
    #Remove smallest marker from original image
    cv2.drawContours(img_data, [contours[min_pos]], -1, 0, 4)
    img[:,:,channel] = img_data
    #get the maximum distance to corner and accomodate the image respectively
    max_dist = np.zeros((2,2,2))
    #up left
    max_dist[0,0,0] = get_e_distance(x, y, 0, 0)
    max_dist[0,0,1] = 0
    #right up
    max_dist[0,1,0] = get_e_distance(x, y, dim[1], 0)
    max_dist[0,1,1] = 1
    #down left
    max_dist[1,0,0] = get_e_distance(x, y, 0, dim[0])
    max_dist[1,0,1] = 2
    #down right
    max_dist[1,1,0] = get_e_distance(x, y, dim[1], dim[0])
    max_dist[1,1,1] = 3

    #check the farthest corner from x and y
    corner = np.argmin(max_dist[:,:,0])
    e_dist = int(np.ceil(np.max(max_dist)))
    #check where to put padding
    x_dir = np.where(max_dist[:,:,1] == corner)[0][0]
    y_dir = np.where(max_dist[:,:,1] == corner)[1][0]
    #how much
    #pad_size_y = int((dim[1]-(2*y)))
    #pad_size_x = int((dim[0]-(2*x)))
    pad_size_y = int((dim[1]-(2*y))/2)
    pad_size_x = int((dim[0]-(2*x))/2)
    if pad_size_y < 0 and pad_size_x < 0:
        pad_size_y = np.abs(pad_size_y)
        pad_size_x = np.abs(pad_size_x)
    if pad_size_y < 0:
        y_dir = 1-y_dir
        pad_size_y = np.abs(pad_size_y)
    if pad_size_x < 0:
        x_dir = 1-x_dir
        pad_size_x = np.abs(pad_size_x)
    #stack zeros to original image
    #create pads
    x_pad = np.full((dim[0], pad_size_x,3), -1)
    y_pad = np.full((pad_size_y, pad_size_x+dim[1], 3), 1)
    if y_dir == 1:
        new_img = np.hstack((img, x_pad))
    if y_dir == 0:
        new_img = np.hstack((x_pad, img))
    if x_dir == 0:
        new_img = np.vstack((new_img, y_pad))
    if x_dir == 1:
        new_img = np.vstack((y_pad, new_img))

    return new_img

def get_polar_projection(data, px_res, return_max = True):
    dim = data.shape
    #projection from image centerpoint
    x = int(dim[0]/2)
    y = int(dim[1]/2)
    if return_max == True:
        r = np.max((x,y))
    else:
        r = np.min((x,y))
    r_res = np.linspace(0, 2 * np.pi, px_res)[:-1]
    #define the pixel resolution i.e. segments in the circle to the final x-axis of the square polar image
    #create a empty template
    polar = np.zeros((r, px_res, 3), dtype = np.float64)
    for ii in range(0,r):
        #for row draw a line from the orginal image center
        for iii in r_res:
            #transform height and width to polar coordinates
            pol_x = ii * np.cos(iii) + x
            pol_y = ii * np.sin(iii) + y
            pol_x = np.clip(pol_x, -1, dim[1]-1)
            pol_y = np.clip(pol_y, -1, dim[0]-1)
            #project each polar coordinate and approximate the position in the final square polar image and each color channel
            polar[ii, int(iii * px_res/2.0/np.pi),0] = data[int(pol_y),int(pol_x),0]
            polar[ii, int(iii * px_res/2.0/np.pi),1] = data[int(pol_y),int(pol_x),1]
            polar[ii, int(iii * px_res/2.0/np.pi),2] = data[int(pol_y),int(pol_x),2]
    return polar

def find_midpoint(data, return_with_nones = False):
    #returns an array of cambium position
    #if not markerd, gives nan
    #go through each column separately
    dim = data.shape
    midpoint = np.zeros((dim[1]))
    #set defaul fill value (center point of the ROI)
    default_fill = int(np.mean(np.where(data > 200)[0]))
    #1st fill with avg value
    midpoint.fill(default_fill)
    for i in range(dim[1]):
        #take column from location data
        a = data[:,i]
        #check if any 255
        a_l = np.where(a == 255)
        #if contains a number take median else insert -1
        if len(a_l[0]) != 0: midpoint[i] = int(np.median(a_l[0]))
        #if position is not specified
        else: midpoint[i] = -1
    if return_with_nones == True:
        #return with empties (-1)
        return midpoint
    else:
        #fill empties with middle point value
        midpoint[np.where(midpoint == -1)] = default_fill
        return midpoint
    
def align_image(data, midpoint, location_ch, signal_ch, cw_ch, return_stat = True):
    dim = data.shape
    location_data = data[:,:,location_ch]
    signal_data = data[:,:,signal_ch]
    cw_data = data[:,:,cw_ch]
    mn = np.min(midpoint)
    mx = np.max(midpoint)
    #get maximun offset
    offset = int(mx - mn)
    #create empty array for aligning columns by midpoint
    aligned_cw = np.zeros((dim[0]+offset,dim[1]), dtype = float)
    aligned_signal = np.zeros((dim[0]+offset,dim[1]), dtype = float)
    aligned_location = np.zeros((dim[0]+offset,dim[1]), dtype = float)
    #create empty array for collecting offset of each column from top
    offset_data = np.zeros((dim[1],1))
    for i in range(0,dim[1]):
        #each column is aligned by alignent point
        point = int(midpoint[i] - mx) * -1
        offset_data[i] = point
        aligned_cw[point:point+dim[0],i] = cw_data[:,i]
        aligned_signal[point:point+dim[0],i] = signal_data[:,i]
        aligned_location[point:point+dim[0],i] = location_data[:,i]
    #Stack images to one image
    #final = np.dstack((aligned_cw, aligned_signal, aligned_location))
    final = np.dstack((aligned_signal, aligned_location, aligned_cw))
    if return_stat == True:
        avg_v_signal = np.mean(aligned_signal, axis = 1)
        avg_v_signal = np.expand_dims(avg_v_signal, axis = 1)
        avg_h_signal = np.mean(aligned_signal, axis = 0)
        avg_h_signal = np.expand_dims(avg_h_signal, axis = 1)
        stats = []
        stats.append(avg_v_signal)
        stats.append(avg_h_signal)
        #append location of marked ROI in aligned image
        loc = int(np.average(np.where(aligned_location == 255)[0]))
        stats.append(loc)
        #append len_y
        stats.append(final.shape[0])
        #append len_x
        stats.append(final.shape[1])
        #append xylem_point/center
        stats.append(offset_data)
        #append periderm/bottom
        last_pos = get_last_px_position(aligned_cw, thresh = 12)
        stats.append(last_pos)
        #append offset position for image i.e. 0
        stats.append(0)
        #append legend
        stats.append(APPENDIX)
        return final, stats
    else: return final

def get_last_px_position(data, thresh = 4):
    dim = data.shape
    #remove empty pixels
    data[data == -1] = 0
    #convert to 8bit
    data = np.uint8(data)
    #resample y by 4
    resized = cv2.resize(data, (dim[1], int(dim[0]/4)), interpolation = cv2.INTER_AREA)
    #convert back to float
    resized = np.float64(resized)
    last_pos = np.zeros((dim[1],1) , dtype = np.float64)
    #go through every column in img
    for i in range(0, dim[1]):
        a = resized[:,i].copy()
        #threshold
        a[a > thresh] = -1
        if np.any(a, -1):
            #if there is any thresholded values get the last index
            pos_index = np.where(a == -1)
            if len(pos_index[0]) == 0:
                last_pos[i] = dim[0]
            else:
                #pos = np.where(a == -1)[0][-1]
                pos = pos_index[0][-1]
                #resize by factor of 4
                pos = pos*4-2
                #save it in array
                last_pos[i] = pos
        else:
            #if no thresholds, take last possible value
            last_pos[i] = dim[0]
    return last_pos

def save_stats(ragged_list, path, fname, add_temp = False):
    #add temp folder
    if add_temp == True: subfolder = '/temp/'
    else: subfolder = ''
    #objectify horrific list
    d = np.asarray(ragged_list, dtype = object)
    #removes extension if any
    fname = splitext(fname)[0]
    #collect path and fname
    file = path + subfolder + fname
    print('Stats saved as: '+file+'.npy')
    #save with numpy and pickle
    np.save(file, d, allow_pickle = True)

def load_stats(path, fname, add_temp = False):
    #add temp folder
    if add_temp == True: subfolder = '/temp/'
    else: subfolder = '/'
    #removes extension if any
    fname = splitext(fname)[0]
    file = path + subfolder + fname + '.npy'
    d = np.load(file, allow_pickle = True)
    return d

def to_rgb_and_save(a, path, fname, suffix = '_projection', return_img = False):
    #convert -1s to 0
    a[np.where(a[:,:,2] == -1)] = 0
    #convert to uint8
    a = np.asarray(a, dtype = np.uint8)
    img = Image.fromarray(a)
    fname = splitext(fname)[0]
    file = path + '/temp/' + fname + suffix + '.png'
    img.save(file)
    if return_img == True: return img
    
def save_img(img, path, fname):
    img = np.asarray(img, dtype = np.uint8)
    img = Image.fromarray(img)
    file = path +'/'+ fname + '.png'
    img.save(file)
    
def assign_dimensions(path, filelist):
    max_y = 0
    loc_over = 0
    loc_under = 0
    for i in enumerate(filelist):
        stats = load_stats(path, i[1])
        #Collect largest y dim
        if stats[3] > max_y:
            max_y = stats[3]
        #Collect largest y - location
        loc_over_temp = stats[3] - stats[2]
        if loc_over_temp > loc_over:
            loc_over = loc_over_temp
        #Collect largest y - (y - location)
        loc_under_temp = stats[3] - loc_over_temp
        if loc_under_temp > loc_under:
            loc_under = loc_under_temp
    #X must be same for all
    x = stats[4]
    return x, max_y, loc_over, loc_under

def draw_ROI(data, line, color, thickness = 1, mode = 'additive'):
    img = data.copy()
    #draws horizontal line on 2d image with specified thicnkess
    thicc = line + thickness
    if mode == 'additive':
        img[line:thicc, :] = img[line:thicc, :] + color
    else:
        img[line:thicc, :] = color
    return img

def draw_line(img, line, color, axis, mode = 'additive'):
    #draws a 1d array line on image
    dim = img.shape
    if axis == 'y':
        for i in range(dim[0]):
            pos = np.round(line[i])
            pos = int(pos)
            if pos >= dim[1]:
                pos = dim[1] -1
            if mode == 'additive':
                img[i, pos] = img[i, pos] + color
            else:
                img[i, pos] = color
    elif axis == 'x':
        for i in range(dim[1]):
            pos = np.round(line[i])
            pos = int(pos)
            if pos >= dim[0]:
                pos = dim[0] -1
            if mode == 'additive':
                img[pos, i] = img[pos, i] + color
            else:
                img[pos, i] = color
                
    #Clip pixel values above 255
    img = np.clip(img, 0, 255)
    return img

def collect_stats(avg_signal, loc, top, down, y_start):
    v_mean = np.average(avg_signal, axis = 1)
    v_mean = np.expand_dims(v_mean, axis = 1)
    h_mean = np.average(avg_signal, axis = 0)
    h_mean = np.expand_dims(h_mean, axis = 1)
    avg_top = np.average(top, axis = 2)
    max_dwn = np.max(down, axis = 2)
    #add padding to location datas
    avg_top = avg_top + y_start
    max_dwn = max_dwn + y_start
            
    #Collect to list
    avg_stats = []
    avg_stats.append(v_mean)
    avg_stats.append(h_mean)
    avg_stats.append(loc)
    avg_stats.append(avg_signal.shape[0])
    avg_stats.append(avg_signal.shape[1])
    avg_stats.append(avg_top)
    avg_stats.append(max_dwn)
    avg_stats.append(y_start)
    avg_stats.append(APPENDIX)
    
    return avg_stats

def align_image_stack(folder, img_files, stat_files, channel = 1):
    #iterate through each
    x, y, loc_o, loc_u = assign_dimensions(folder, stat_files)
    z = len(img_files)
    #Empty array for collecting all 2d signals/images
    signal_stack = np.zeros(((loc_o+loc_u), x, z), dtype = np.float64)
    #Empty array for collecting center/top
    top = np.zeros((x, 1, z), dtype = np.float64)
    #Empty array for collecting perider/down
    dwn = np.zeros((x, 1, z), dtype = np.float64)
    #Open image an image stats respectively
    for i in enumerate(zip(img_files, stat_files)):
        #load stats
        stats = load_stats(folder, i[1][1])
        #load image as np
        data = np.array(Image.open(folder+'/'+i[1][0]), dtype = np.float64)
        #load it in 3d array
        y_start = loc_u - stats[2]
        y_end = y_start + data.shape[0]
        signal_stack[y_start:y_end, :, i[0]] = data[:, :, channel]
        #load top    to_rgb_and_save(final, i[1], 'avg_signal')
        top[:, 0, i[0]] = stats[5].ravel()
        #load dwn
        dwn[:, 0, i[0]] = stats[6].ravel()
        #Collect averages
    avg_signal = np.average(signal_stack, axis = 2)
    avg_stats = collect_stats(avg_signal, loc_u, top, dwn, y_start)
    
    return signal_stack, avg_stats


def show_stack(stack, mode = 'gray'):
    if mode == 'gray':
        img_n = stack.shape[2]
        fig = plt.figure()
        for i in range(0, img_n):
            ax1 = fig.add_subplot(1,img_n+1 , i+1)
            ax1.imshow(stack[:,:,i])
    if mode == 'rbg':
        img_n = stack.shape[3]
        fig = plt.figure()
        for i in range(0, img_n):
            ax1 = fig.add_subplot(1,img_n+1 , i+1)
            ax1.imshow(stack[:,:,:,i])        

def rolling_median(x, window = 3):
    k = (window-1)/2
    x_conv = np.zeros(len(x))
    kernel = np.arange(-k,k,1)
    conv = np.zeros(window)
    for i in range(0, len(x)):
        cnt = 0
        for ii in kernel:
            pos = int(i+ii)
            if pos >= len(x):
                pos = pos - len(x)
            conv[cnt] = x[pos]
            cnt += 1
        x_conv[i] = np.median(conv)
    return x_conv

def create_legend_box(dim, size, color = 255):
    inv_color = 255 - color
    x = int(dim/4)
    a = np.zeros((size, int(x/2)), dtype = np.uint8)
    b = np.full((size, x), color, dtype = np.uint8)
    c = np.zeros((size, x), dtype = np.uint8)
        
    b = cv2.putText(b, text = 'PhloemP', org=(int(b.shape[1]*2/8), int(b.shape[0]*3/5)),
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale = 1,
                color=(inv_color, inv_color, inv_color), thickness = 1)
    c = cv2.putText(c, text = 'XPP', org=(int(b.shape[1]*2/8), int(b.shape[0]*3/5)),
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale = 1,
                color=(color, color, color), thickness = 1)
    
    chan = np.hstack((a, b , c, b, a))
    final = np.dstack((chan, chan, chan))
    return final
    
def create_avg_signal(xy, size, data, offset, orientation = 'v', cmap = 'viridis'):
    data = data.ravel()
    if orientation == 'v':
        #add offset to start of data  if its positive
        if offset > 0: data = np.concatenate((np.zeros(offset), data))
        #add filling to match length if rmeainder is positive
        if not xy-len(data) <= 0:
            rest = np.zeros(xy-len(data))
            data = np.concatenate((data, rest))
        #Create vertical signal
        signal = np.zeros((xy, size), dtype = np.float64)
        for i in range(0, size):
            signal[:,i] = data
    elif orientation == 'h':
        #Create vertical signal
        signal = np.zeros((xy, size), dtype = np.float64)
        for i in range(0, size):
            signal[:,i] = data
        signal = np.transpose(signal)
    #square transform
    signal = signal * signal
    #scale to 255
    signal = signal * (255/np.max(signal))
    signal = np.array(signal, dtype = np.uint8)
    #final = np.dstack((signal, signal, signal))
    final = cmapy.colorize(signal, cmap)
    return final

def create_title_box(x, size, title, color = 255):
    title = splitext(str(title))[0]
    pos_x = int((x * 1/2) - (len(title)-1)*5.5)
    if pos_x < 0: pos_x = 0
    pos_y = int(size * 6/8)
    a = np.zeros((size, x, 3), dtype = np.uint8)
    a = cv2.putText(a, text = title, org=(pos_x, pos_y),
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale = 1.2,
                color=(color, color, color), thickness = 1)
    
    return a

def create_avg_signal(xy, size, data, offset, orientation = 'v', cmap = 'gray'):
    if orientation == 'v':
        #add offset to start of data
        data = np.concatenate((np.zeros(offset), data.ravel()))
        #add filling to match length
        rest = np.zeros(xy-len(data))
        data = np.concatenate((data, rest))
        #Create vertical signal
        signal = np.zeros((xy, size), dtype = np.float64)
        for i in range(0, size):
            signal[:,i] = data
    elif orientation == 'h':
        #Create vertical signal
        signal = np.zeros((xy, size), dtype = np.float64)
        for i in range(0, size):
            signal[:,i] = data.ravel()
        signal = np.transpose(signal)
    #square transform
    signal = signal * signal
    #scale to 255
    signal = signal * (255/np.max(signal))
    final = np.dstack((signal, signal, signal))
    return final

def add_block(img, add_on, position = 'bottom'):
    if position == 'bottom':
        final = np.vstack((img, add_on))
    elif position == 'top':
        final = np.vstack((add_on, img))
    elif position == 'side':
        final = np.hstack((img, add_on))
    else:
        print('Position: bottom, top or side')
        return
    return final