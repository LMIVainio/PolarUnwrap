

paths = ['/your list of folders',
         '',
         '']

#paths =['/home/leovain/Desktop/Desktop/Polar/test']
p_path = str(Path(paths[0]).parent)

PREFIX = 'yourprefix_'
PHI = 480
SIGNAL_DATA = 0
LOC_DATA = 1
CW_DATA = 2

#Make projections of each path
for i in enumerate(paths):
    #Create a folder for temporary image files
    if not exists(i[1]+'/temp'):
        makedirs(i[1]+'/temp')
        print('Temp folder created')
    #Get all same type files from folder
    img_files = get_filelist(i[1], '.png')
    for ii in enumerate(img_files):
        #Open image as np array
        data = np.array(Image.open(i[1]+'/'+ii[1]), dtype = np.uint8)
        #Create padding for centering
        centered = pad_image_from_centerpoint(data, channel = LOC_DATA)
        #Project to polar coordinates with sampling rate of phi
        polar = get_polar_projection(centered, px_res = PHI)
        #get array of annotated ROI
        midpoint = find_midpoint(polar[:,:,LOC_DATA])
        #Align image with midpoint array
        #Function gives statistics of ROI location also
        aligned_img, stats = align_image(polar, midpoint, location_ch = LOC_DATA, signal_ch = SIGNAL_DATA, cw_ch = CW_DATA)
        #save stats as numpy
        save_stats(stats, i[1], ii[1], add_temp = True)
        to_rgb_and_save(aligned_img, i[1], ii[1])
#Create list for each average image stats
stats_master = []
#Collect averages of each path
for i in enumerate(paths):
    #load filelists for images and stats
    img_files = get_filelist(i[1]+'/temp', 'projection.png')
    stat_files = get_filelist(i[1]+'/temp', '.npy')
    #Align projections
    signal_stack, avg_stats = align_image_stack(i[1]+'/temp', img_files, stat_files, channel = 0)
    stats_master.append(avg_stats)
    #get average of signal stack
    avg_signal = np.average(signal_stack, axis = 2)
    final = np.dstack((avg_signal, avg_signal, avg_signal))
    final = np.asarray(final, dtype = np.uint8)
    #create nme from folder
    final_name = PREFIX + i[1].replace(p_path + '/', "", 1) + '_avg'
    save_img(final, p_path, final_name)
    save_stats(avg_stats, p_path + '/', final_name, add_temp = False)

#Align average projections
#load filelists for images and stats from parent directory
img_files = get_filelist(p_path, '_avg.png')
stat_files = get_filelist(p_path, '.npy', )
signal_stack, avg_stats = align_image_stack(p_path, img_files, stat_files)
dim = signal_stack.shape


#Annotate avg_images
#Variable for setting drawing offset
offset_add = 0
#Create legend for location data
legend_size = 20
leg_pos = 'bottom'
if leg_pos == 'top': offset_add = offset_add + legend_size
#Create empty title
tit_size = 25
tit_pos = 'top'
if tit_pos == 'top': offset_add = offset_add + tit_size
#horizontal row size
hor_size = 8
#vertival column size
ver_size = 8
#create empty array for all images
canvas = np.zeros((dim[0] + legend_size + tit_size + ver_size, dim[1] + hor_size, 3, dim[2]), dtype = np.uint8)
#spacer between images
spacer_size = 8
spacer = np.full((dim[0] + legend_size + tit_size + hor_size, spacer_size, 3), 255)
#create empty array for stiching images horizontally
panorama = np.zeros((dim[0] + legend_size + tit_size + hor_size, (signal_stack.shape[1] + spacer_size + hor_size)*signal_stack.shape[2], 3))
#Normalize globally
mx = np.max(signal_stack)
signal_stack = signal_stack/mx*255
#Draw stats
for i in range(0, len(img_files)):
    #Draw top in red
    r = draw_line(signal_stack[:,:,i].copy(), stats_master[i][5], color = 120, axis = 'x')
    #Draw ROI
    g = draw_ROI(signal_stack[:,:,i].copy(), avg_stats[2], color = 120)
    #Draw dwon in blue
    dwn = stats_master[i][6]
    dwn = rolling_median(dwn, window = 12)
    b = draw_line(signal_stack[:,:,i].copy(), dwn, color = 255, axis = 'x')
    rgb = np.dstack((r,g,b))
    rg = 
    #Add positional explanation
    legend_box = create_legend_box(dim[1], size = legend_size, color = 240)
    rgb = add_block(rgb, legend_box, position = leg_pos)
    #Add title
    title_box = create_title_box(dim[1], size = tit_size, title = img_files[i])
    rgb = add_block(rgb, title_box, position = tit_pos)
    #Add average horizontal line for vertical signal
    h_avg =  np.average(signal_stack[:,:,i], axis = 0)
    h_sig_block = create_avg_signal(rgb.shape[1], hor_size, h_avg, offset = 0, orientation = 'h', cmap = 'gnuplot2')
    rgb = add_block(rgb, h_sig_block, position = 'bottom')
    #Add average vertical line for horizontal signal
    v_avg = np.average(signal_stack[:,:,i], axis = 1)
    v_sig_block = create_avg_signal(rgb.shape[0], ver_size, v_avg, offset = offset_add, orientation = 'v', cmap = 'gnuplot2')
    rgb = add_block(rgb, v_sig_block, position = 'side')
    canvas[:,:,:,i] = rgb
    #if spacer_size != 0: rgb = np.hstack((spacer, rgb))
    x_start = i*rgb.shape[1] + (i+1)*spacer_size
    x_end = ((i+1)*rgb.shape[1] + (i+1)*spacer_size)
    #if x_end >= panorama.shape[1]: x_end = panorama.shape[1]-1
    panorama[:,x_start:x_end,:] = rgb
#slice 1st spacer off of panorama
panorama = panorama[:, spacer_size:-1, :]
#convert to 8-bit
panorama = np.asarray(panorama, dtype = np.uint8)
save_img(panorama, p_path, 'panorama')
for i in range(0, dim[2]):
    save_img(canvas[:,:,:,i], p_path, 'final_'+str(i))


