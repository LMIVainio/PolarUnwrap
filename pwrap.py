##Import packages
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir, makedirs, path
from os.path import isfile, join, dirname, exists, splitext
from pathlib import Path
from matplotlib.pyplot import imsave
import cv2
import cmapy

global APPENDIX
APPENDIX = ['[0] v_signal', '[1] h_signal', '[2] ROI location', '[3] y or height', '[4] x or width', '[5] center or top', '[6] periderm or down', '[7] offset']

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

def pad_image_from_centerpoint(img_array, channel = 1, fill = 0):
    #creates an image with equal distance to x and y edges from predetermined center point
    #duplicate data
    img = img_array.copy()
    img.setflags(write = True)
    #convert -1s to 0's
    img[img_array == -1] = 0
    #take unsigned integer from location data channel
    location_data = img[:,:,channel].astype('uint8')
    dim = location_data.shape
    center_point = np.zeros(dim, dtype = np.uint8)
    #remove 255 and take only 253 or centerpoint data
    center_point[location_data == 255] = 0
    center_point[location_data == 253] = 255
    #threshold the positional information
    ret, thresh = cv2.threshold(center_point, 252, 255, 0)
    #get contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #get countour ares
    areas = get_contour_areas(contours)
    if len(areas) == 0:
        print('Unmarked image found')
        return img
    #get smallest area contour position
    min_pos = np.argmin(areas)
    #warn if center dot is too large or only one area found
    if areas[min_pos] > 500 and len(areas) == 1:
        print('Only one marker area found and it is too big, assuming middle pixel as middlepoint')
        x = int(int(dim[0]/2))
        y = int(int(dim[1]/2))
    else:
        M = cv2.moments(contours[min_pos])
        #get the centerpoint of the smallest area
        if M["m00"] == 0:
            print('Smallest marker area too small or non existing, assuming middle pixel as middlepoint')
            x = int(int(dim[1]/2))
            y = int(int(dim[0]/2))
        else:
            x_c = int(M["m10"] / M["m00"])
            y_c = int(M["m01"] / M["m00"])
            x,y,w,h = cv2.boundingRect(contours[min_pos])
            x = int(np.round(x+(w/2)))
            y = int(np.round(x+(h/2)))
    #Remove smallest marker from original image
    cv2.drawContours(location_data, [contours[min_pos]], -1, 0, 4)
    img[:,:,channel] = location_data
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
    #create pads and fill the with fill = 0
    x_pad = np.full((dim[0], pad_size_x,3), fill)
    y_pad = np.full((pad_size_y, pad_size_x+dim[1], 3), fill)
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
    ##old version of the function
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

def project_to_polar(data, px_res, return_visualization = False):
    #projects a 2d with 3 channel to polar with sampling resolution of px_res
    dim = data.shape
    #get center
    center_y = int(dim[0]/2)
    center_x = int(dim[1]/2)
    half_point = np.min((center_x, center_y))
    #generate empty array for visualizing sampling resolution and establish running
    #number for filling the array
    sampling_vis = np.zeros((dim[0], dim[1], 2), dtype = np.float64)
    id_runner = 0
    #generate and array for sampling ray half of image min(width,height)
    r = np.arange(0, half_point)
    #generate and array of phi values with sampling frequency tau/px_res
    phi = np.linspace(0, 2*np.pi, px_res)
    #define the pixel resolution i.e. segments in the circle to the final x-axis of the square polar image
    #create a empty template
    polar = np.zeros((len(r), len(phi), 3), dtype = np.float64)
    #loop through the polar projection pixels and fill it one by one by sampling the carthesian coordinate image
    for p_x, phi_i in enumerate(phi):
        #p_x is the phi value at p_x
        for p_y, r_i in enumerate(r):
            #p_y is the radius at r_i
            #polar conversion is x = r*cos(phi) and y = r*sin(phi)
            sample_coordinate_x = int(r_i*np.cos(phi_i)+center_x)
            sample_coordinate_y = int(r_i*np.sin(phi_i)+center_y)
            #make sure that the sampling does not go over the edge of image of origin
            if sample_coordinate_x > dim[1]:
                continue
            if sample_coordinate_y > dim[0]:
                continue
            #fill polar projection array and sample original image with (int) approximation of each coordinate
            polar[p_y,p_x,0] = data[sample_coordinate_y,sample_coordinate_x,0]
            polar[p_y,p_x,1] = data[sample_coordinate_y,sample_coordinate_x,1]
            polar[p_y,p_x,2] = data[sample_coordinate_y,sample_coordinate_x,2]
            #fill visualization image
            #fills array with a running number from the sampling site of the original image
            sampling_vis[sample_coordinate_y,sample_coordinate_x,0] = id_runner
            sampling_vis[sample_coordinate_y,sample_coordinate_x,1] = 1
            id_runner += 1
        id_runner += 1
    if return_visualization == True:
        return polar, sampling_vis
    else:
        return polar
    
def project_back_from_polar(polar, dim, px_res):
    #projects polar image back to shape
    #NOTE works for only 1 channel images
    #Create an empty array for drawing a polar projection
    back_projection = np.zeros((dim[0],dim[1]), np.uint8)
    #set up original image center point for backprojection
    center_y = int(dim[0]/2)
    center_x = int(dim[1]/2)
    half_point = np.min((center_x, center_y))
    #generate and array for sampling ray half of image min(width,height)
    r = np.arange(0, half_point)
    #generate and array of phi values with sampling frequency tau/px_res
    phi = np.linspace(0, 2*np.pi, px_res)
    #iterate through each pixel in polar projection
    for p_x, phi_i in enumerate(phi):
        #same as above
        for p_y, r_i in enumerate(r):
            #same as above, loop through eah pixel of the image
            drawing_coordinate_x = int(r_i*np.cos(phi_i)+center_x)
            drawing_coordinate_y = int(r_i*np.sin(phi_i)+center_y)
            #make sure that drawing does not happen over borders
            if drawing_coordinate_x > dim[1]:
                continue
            if drawing_coordinate_y > dim[1]:
                continue
            #pixel_by pixel fill the back_projection
            back_projection[drawing_coordinate_y,drawing_coordinate_x] = polar[p_y,p_x]
    return back_projection

def find_midpoint(data, return_with_nones = False):
    #returns an array of cambium position
    #if not markerd, gives nan
    #go through each column separately
    dim = data.shape
    midpoint = np.zeros((dim[1]))
    #set defaul fill value (center point of the ROI)
    try:
        default_fill = int(np.mean(np.where(data > 200)[0]))
    except:
        default_fill = 0
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
    
def align_image(data, midpoint, location_ch, signal_ch, cw_ch, guess_cw_position = False, return_stat = True):
    #aligns image columns with midpoint 1d array
    #get image dimensions
    dim = data.shape
    #set location data channel
    location_data = data[:,:,location_ch]
    #set signal data channel
    signal_data = data[:,:,signal_ch]
    #set cell wall data channel
    cw_data = data[:,:,cw_ch]
    #find minmax of midpoint 1d array
    mn = np.min(midpoint)
    mx = np.max(midpoint)
    #get maximun offset
    offset = int(mx - mn)
    #get a median from midpointdata (if is data omitted, use median)
    #1st remove -1s if any
    m = np.where(midpoint != -1)
    try:
        m = int(np.median(midpoint))
    except:
        m = 0
    #create empty array for aligning columns by midpoint
    aligned_cw = np.zeros((dim[0]+offset,dim[1]), dtype = float)
    aligned_signal = np.zeros((dim[0]+offset,dim[1]), dtype = float)
    aligned_location = np.zeros((dim[0]+offset,dim[1]), dtype = float)
    #create empty array for collecting offset of each column from top
    offset_data = np.zeros((dim[1],1))
    #create blank array for omitting signal data
    blank = np.zeros((signal_data[:,0].shape))
    for i in range(0,dim[1]):
        #each column is aligned by aligment point
        point = int(midpoint[i] - mx) * -1
        m_point = int(m - mx) * -1
        offset_data[i] = point
        #In case of not defined midpoint in data
        if midpoint[i] == -1: 
            aligned_signal[point:point+dim[0],i] = blank
            #Guess the position of cw
            if guess_cw_position == True: 
                aligned_cw[m_point:m_point+dim[0],i] = cw_data[:,i]
                offset_data[i] = m_point
            #don't guess and set cell wall and center to 0-point
            else: 
                #aligned_cw[0:len(cw_data[:,i]),i] = cw_data[:,i]
                offset_data[i] = 0
        else:
            aligned_signal[point:point+dim[0],i] = signal_data[:,i]
            aligned_cw[point:point+dim[0],i] = cw_data[:,i]
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
        try:
            loc = int(np.average(np.where(aligned_location == 255)[0]))
        except:
            loc = 0
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

def save_stats(ragged_list, filepath, fname, add_temp = False):
    #add temp folder
    if add_temp == True: subfolder = '/temp/'
    else: subfolder = ''
    #objectify horrific list
    d = np.asarray(ragged_list, dtype = object)
    #removes extension if any
    fname = splitext(fname)[0]
    #collect path and fname
    file = filepath + subfolder + fname
    print('Stats saved as: '+file+'.npy')
    #save with numpy and pickle
    np.save(file, d, allow_pickle = True)

def load_stats(filepath, fname, add_temp = False):
    #add temp folder
    if add_temp == True: subfolder = '/temp/'
    else: subfolder = '/'
    #removes extension if any
    fname = splitext(fname)[0]
    file = filepath + subfolder + fname + '.npy'
    d = np.load(file, allow_pickle = True)
    return d

def to_rgb_and_save(a, filepath, fname, suffix = '_projection', return_img = False):
    #convert -1s to 0
    a[np.where(a[:,:,2] == -1)] = 0
    #convert to uint8
    a = np.asarray(a, dtype = np.uint8)
    img = Image.fromarray(a)
    fname = splitext(fname)[0]
    file = filepath + '/temp/' + fname + suffix + '.png'
    img.save(file)
    if return_img == True: return img
    
def save_img(img, filepath, fname):
    img = np.asarray(img, dtype = np.uint8)
    img = Image.fromarray(img)
    file = filepath +'/'+ fname + '.png'
    img.save(file)
    
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

def assign_dimensions(filepath, filelist):
    max_y = 0
    loc_over = 0
    loc_under = 0
    if len(filelist) == 0:
        print('empty folder in: '+filepath)
        return
    for i in enumerate(filelist):
        stats = load_stats(filepath, i[1])
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
    signal = np.asarray(signal, dtype = np.uint8)
    #final = np.dstack((signal, signal, signal))
    final = cmapy.colorize(signal, cmap, rgb_order = True)
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

def save_stats_csv(stat_array, name_list, filepath, stat_select = 0):
    #Save average master stats to a csv
    #check if there are unexpected files in folder
    if len(stat_array) != len(name_list):
        print("Discrepancy in file amount while saving stats")
        #return
    #1st go through stats for generating empty dataFrame
    #save largest numbers
    loc_m = 0
    d_stat_m = 0
    for i in range(0,len(stat_array)):
        #ROI location is fixed to 3rd position
        loc = stat_array[i][2]
        #check if it is larger than saved value
        if loc > loc_m: loc_m = loc
        #check the length of stat - loc of interest
        d_stat = len(stat_array[i][stat_select]) + loc
        #check that stat of interest is longer thatn saved value
        if d_stat > d_stat_m: d_stat_m = d_stat
    #The final array cannot be longer than loc_m+d_stat_m      
    #create emtpy array for collecting all stats
    A = np.zeros((d_stat_m+loc_m, len(stat_array)), dtype = np.float32)
    A.fill(-1)
    #collect stats to A
    for i in range(0,len(stat_array)):
        loc = stat_array[i][2]
        a  = stat_array[i][stat_select]
        point = loc_m - loc
        A[point:point+len(a), i] = a.ravel()
    #Generate axis with 0 marking cambium
    negs = np.arange(-loc_m,0)
    posi = np.arange(0,d_stat_m)
    labels = np.zeros(len(negs)+len(posi), dtype = np.uint16)
    labels[0:len(negs)] = abs(negs)
    labels[len(negs):] = posi
    #depending on version, this migh be required for making labels 1 thick
    #labels = np.expand_dims(labels, axis = 1)
    #A = np.hstack((labels, A))
    #generate labels
    n = []
    for name in name_list:
        #take filename without file-extension
        name = splitext(name)[0]
        n.append(name)
    #remove nonfilled rows
    A_sum = np.sum(A, axis = 1)
    A = A[A_sum != -3, :]
    labels = labels[:A.shape[0]]
    #create pandas dataframe
    df = pd.DataFrame(A, index = labels, columns = n)
    fname = str(path.basename(filepath)) + '_statsfile.csv'
    df.to_csv(filepath+'/'+fname, sep = ';')
    #return A