#GUI packages
import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter import messagebox, Entry
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
import pwrap as p

#Define script functions


class MyGUI:
    #Defining main
    def __init__(self, master):
        #Setting up GUI
        self.master = master
        master.title("Polar unwrap simpleGUI")

        # Create a label for the text prompt
        self.label_prompt = tk.Label(master, text="Please select a folder:")
        self.label_prompt.grid(row=0, column=0, sticky="w", padx=10, pady=10)

        # Create a button to open the folder dialog
        self.button_open = tk.Button(master, text="Open Folder", command=self.open_file)
        self.button_open.grid(row=0, column=1, padx=10, pady=10)
        
        # Create a button for removein last folder
        self.button_remove = tk.Button(master, text = 'Remove Folder', command = self.remove_folder)
        self.button_remove.grid(row = 1, column = 0, padx = 10, pady = 10)
        
        # Create a check box for interpolation
        self.check_interpolation = tk.BooleanVar()
        self.check_button_interpolation = tk.Checkbutton(master, text="Use interpolation", variable=self.check_interpolation)
        self.check_button_interpolation.grid(row=1, column=1, sticky="w", padx=10, pady=10)
        
        # Create a check box for overwrite
        self.check_overwrite = tk.BooleanVar()
        self.check_button_overwrite = tk.Checkbutton(master, text="Overwrite old projected files", variable=self.check_overwrite)
        self.check_button_overwrite.grid(row=2, column=1, sticky="w", padx=10, pady=10)

        # Create a label to display the selected file
        self.label_file = tk.Label(master, text="")
        self.label_file.grid(row=2, column=0, sticky="w", padx=10, pady=10)
        
        #Create a label to display PHI
        self.label_PHI = tk.Label(master, text="Set PHI")
        self.label_PHI.grid(row = 3, column = 0, sticky = 'w', padx = 10, pady = 10)
        
        # Create an input field for PHI
        self.textbox_PHI = Entry(master)
        self.textbox_PHI.grid(row = 4, column = 0)
        self.textbox_PHI.insert(0, '1024')
        
        #Create a label for displaying filename prefix
        self.label_prefix = tk.Label(master, text = 'Filename prefix:')
        self.label_prefix.grid(row = 3, column = 1, sticky = 'w', padx = 10, pady = 10)
        
        # Create an input field for filename prefix
        self.textbox_PREFIX = Entry(master)
        self.textbox_PREFIX.grid(row = 4, column = 1)

        # Create a button to run script
        self.button_add = tk.Button(master, text="Run polar unwrap", command=self.run_polar_unwrap)
        self.button_add.grid(row=8, column=0, padx=10, pady=10)

        # Create a button to quit the program
        self.button_quit = tk.Button(master, text="Quit", command=master.quit)
        self.button_quit.grid(row=8, column=1, padx=10, pady=10)
        
    ##Defining GUI functions
    
    def open_file(self):
        # Open a file dialog to select a file
        home_dir = '/home'
        dir_name = filedialog.askdirectory() 
        if dir_name:
            # If there is a folder get it and add the new one
            paths = self.label_file.cget("text").split('\n')
            old_folder = self.label_file.cget("text")
            new_folders = old_folder+'\n'+dir_name
            if dir_name not in paths:
                self.label_file.config(text=new_folders)
            else:
                messagebox.showinfo("Duplicate error", "The folder is already added")
                
    def remove_folder(self):
        #get paths string
        paths = self.label_file.cget('text')
        new_line_i = paths.rfind('\n')
        sub_s = paths[0:new_line_i]
        self.label_file.config(text = sub_s)

    def run_polar_unwrap(self):
        #get paths as list
        paths = self.label_file.cget("text").split('\n')
        paths.remove('')
        #Get file prefix for saving
        PREFIX = self.textbox_PREFIX.get()
        #Define the sampling density
        PHI = int(self.textbox_PHI.get())
        #define the image data channel for signal
        SIGNAL_DATA = 0
        #positional information channel in the image
        LOC_DATA = 1
        #cell wall channel in the image
        CW_DATA = 2
        #Omit nonmarked data i.e.
        #do not use the iinterpolation in case of missing positional information (True/False)
        ONMD = self.check_interpolation.get()
        #switch it (this should be changed)
        if not ONMD: ONMD = True
        #Overwrite temp folder
        OWT = self.check_overwrite.get()
        #Set image type
        if paths == ['']:
            messagebox.showinfo("Error",'No folders selected')
        img_files = p.get_filelist(paths[0], '.png')
        ##DEBUG
        #messagebox.showinfo("Debug","Paths:"+str(paths)+'\nPrefix: '+PREFIX+'\nPHI: '+str(PHI)+'\ninterpolation: '+str(ONMD)+'\noverwrite: '+str(OWT)+'\nimg_files: '+str(img_files))
        ##DEBUG
        
        ##Start unwrapping
        
        #get parent folder name
        p_pathname = str(path.basename(paths[0]))
        #get parent dir  if any
        if len(paths) == 1:
            if not exists(paths[0]+'/results'):
                makedirs(paths[0]+'/results')
                print('Results folder created')
            p_path = paths[0]+'/results'
        else: p_path = str(Path(paths[0]).parent)


        #Make projections of each path
        for i in enumerate(paths):
            #Create a folder for temporary image files
            if not exists(i[1]+'/temp'):
                makedirs(i[1]+'/temp')
                print('Temp folder created')
            elif OWT == False: continue
            #Get all same type files from folder
            img_files = p.get_filelist(i[1], '.png')
            for ii in enumerate(img_files):
                #Open image as np array
                data = np.array(Image.open(i[1]+'/'+ii[1]), dtype = np.uint8)
                #Create padding for centering
                centered = p.pad_image_from_centerpoint(data, channel = LOC_DATA)
                #Project to polar coordinates with sampling rate of phi
                polar = p.project_to_polar(centered, px_res = PHI)
                #get array of annotated ROI
                midpoint = p.find_midpoint(polar[:,:,LOC_DATA], return_with_nones = ONMD)
                #Align image with midpoint array
                #Function gives statistics of ROI location also
                aligned_img, stats = p.align_image(polar, midpoint, location_ch = LOC_DATA, signal_ch = SIGNAL_DATA, cw_ch = CW_DATA)
                #save stats as numpy
                p.save_stats(stats, i[1], ii[1], add_temp = True)
                p.to_rgb_and_save(aligned_img, i[1], ii[1])
        #Create list for each average image stats
        stats_master = []
        #Collect averages of each path
        for i in enumerate(paths):
            #load filelists for images and stats
            img_files = p.get_filelist(i[1]+'/temp', 'projection.png')
            stat_files = p.get_filelist(i[1]+'/temp', '.npy')
            #Align projections
            signal_stack, avg_stats = p.align_image_stack(i[1]+'/temp', img_files, stat_files, channel = 0)
            stats_master.append(avg_stats)
            #get average of signal stack
            avg_signal = np.average(signal_stack, axis = 2)
            final = np.dstack((avg_signal, avg_signal, avg_signal))
            final = np.asarray(final, dtype = np.uint8)
            #create nme from folder if there is a parent
            p_pathname = str(path.basename(i[1]))
            final_name = PREFIX + p_pathname + '_avg'
            print(final_name)
            p.save_img(final, p_path + '/', final_name)
            p.save_stats(avg_stats, p_path + '/', final_name, add_temp = False)

        #Align average projections
        #load filelists for images and stats from parent directory
        img_files = p.get_filelist(p_path, '_avg.png')
        stat_files = p.get_filelist(p_path, '.npy', )
        signal_stack, avg_stats = p.align_image_stack(p_path, img_files, stat_files)
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
        hor_size = 20
        #vertival column size
        ver_size = 20
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
            r = p.draw_line(signal_stack[:,:,i].copy(), stats_master[i][5], color = 120, axis = 'x')
            #Draw ROI
            g = p.draw_ROI(signal_stack[:,:,i].copy(), avg_stats[2], color = 120)
            #Draw dwon in blue
            dwn = stats_master[i][6]
            dwn = p.rolling_median(dwn, window = 12)
            b = p.draw_line(signal_stack[:,:,i].copy(), dwn, color = 255, axis = 'x')
            rgb = np.dstack((r,g,b))
            #Add positional explanation
            legend_box = p.create_legend_box(dim[1], size = legend_size, color = 240)
            rgb = p.add_block(rgb, legend_box, position = leg_pos)
            #Add title
            title_box = p.create_title_box(dim[1], size = tit_size, title = img_files[i])
            rgb = p.add_block(rgb, title_box, position = tit_pos)
            #Add average horizontal line for vertical signal
            h_avg =  np.average(signal_stack[:,:,i], axis = 0)
            h_sig_block = p.create_avg_signal(rgb.shape[1], hor_size, h_avg, offset = 0, orientation = 'h', cmap = 'gnuplot2')
            rgb = p.add_block(rgb, h_sig_block, position = 'bottom')
            #Add average vertical line for horizontal signal
            v_avg = np.average(signal_stack[:,:,i], axis = 1)
            v_sig_block = p.create_avg_signal(rgb.shape[0], ver_size, v_avg, offset = offset_add, orientation = 'v', cmap = 'gnuplot2')
            rgb = p.add_block(rgb, v_sig_block, position = 'side')
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
        p.save_img(panorama, p_path, 'panorama')
        for indx in range(0, dim[2]):
            p.save_img(canvas[:,:,:,i], p_path, 'final_'+str(indx))
        #write stats to csv
        p.save_stats_csv(stats_master, img_files,  p_path, stat_select = 0)
        #show panorama
        plt.imshow(panorama)

        
        
        
# Create the main window
root = tk.Tk()

# Create the GUI
my_gui = MyGUI(root)

# Run the GUI loop
root.mainloop()