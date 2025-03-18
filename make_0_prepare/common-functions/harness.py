import numpy as np
# from scipy.signal import kaiserord, lfilter, firwin, freqz
# from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
# from joblib import Parallel, delayed
from skimage.feature import hog
import pickle

import numpy as np
import matplotlib.pyplot as plt

def save_pickle(model,pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(pkl_filename):
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model

from skimage.draw import polygon
import scipy.ndimage as ndimage
def contour_to_roi(image, contour):
    r_mask = np.zeros_like(image, dtype=bool)

    # Extract x, y coordinates and close the contour if necessary
    x = np.round(contour[:, 0]).astype(int)
    y = np.round(contour[:, 1]).astype(int)

    # Ensure valid indices
    x = np.clip(x, 0, image.shape[0] - 1)
    y = np.clip(y, 0, image.shape[1] - 1)

    # Fill the polygon using skimage's `polygon` function
    rr, cc = polygon(x, y, image.shape)
    r_mask[rr, cc] = 1

    # Fill holes inside the region
    r_mask = ndimage.binary_fill_holes(r_mask)

    # Convert to float and set background to NaN
    r_mask = r_mask.astype(float)
    r_mask[r_mask == 0] = np.nan

    return r_mask


def show_map(data,caxis,if_return_map=False,cmap='rocket',IDX_sorted=[]):
    ss_idx = IDX_sorted.shape;
    IDX_reshape = IDX_sorted.reshape((-1,1))
    tmp = np.empty((ss_idx[0]*ss_idx[1],1))
    tmp[:] = np.nan
    std_map = tmp
    ncomps=500
    for i_region in range(ncomps):
        indice = IDX_reshape == i_region+1
        std_map[indice] =  data[i_region]
    # reshape back
    std_map = np.reshape(std_map,(ss_idx[0],ss_idx[1]))

    if if_return_map ==True:
        return std_map
    else:
        plt.imshow(std_map,vmin=caxis[0],vmax=caxis[1],cmap=cmap)
        plt.axis('off')
        plt.colorbar()

#First we should filter input_array so that it does not contain NaN or Inf.
def normalize(some_data):
    input_array=np.array(some_data)
    if np.unique(input_array).shape[0]==1:
        pass #do thing if the input_array is constant
    else:
        result_array=(input_array-np.min(input_array))/np.ptp(input_array)
    #To extend it to higher dimension, add axis= kwarvg to np.min and np.ptp
    return result_array


def start_stop(a, trigger_val):
    # "Enclose" mask with sentients to catch shifts later on
    mask = np.r_[False,np.equal(a, trigger_val),False]

    # Get the shifting indices
    idx = np.flatnonzero(mask[1:] != mask[:-1])

    # Get the start and end indices with slicing along the shifting ones
    #return zip(idx[::2], idx[1::2]-1)
    return np.reshape(idx,(-1,2))

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def IsPointInROI(x1, y1, x2, y2, x, y) : 
    if (x > x1 and x < x2 and y > y1 and y < y2) : 
        return True
    else: 
        return False
# import subprocess
# def deflicker(video_path,output_path):
#     command = 'ffmpeg -fflags +genpts -i {video} -fflags +genpts -i {video} -filter_complex "[0:v]setpts=PTS-STARTPTS[top]; [1:v]setpts=PTS-STARTPTS+.033/TB, format=yuva420p, colorchannelmixer=aa=0.5[bottom]; [top][bottom]overlay=shortest=1" -c:v libx264 -crf 26 -an {output}'.format(
#         video=video,
#         output=output)
#     subprocess.call(command,shell=True)



# import cv2
# import numpy as np

# def convert_vid_to_matrix(filename,dsf=1):
#     cap = cv2.VideoCapture(filename)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     while not cap.isOpened():
#         cap = cv2.VideoCapture(filename)
#         cv2.waitKey(1000)
#         print("Wait for the header")
    
#     pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
#     for i in range(0,total_frames):
#         flag, frame = cap.read()
#         if flag:
#             # The frame is ready and already captured
#             #cv2.imshow('video', frame)
#             if i > 0:
#                 im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 gray[i,:,:] = cv2.resize(im,(int(im.shape[1]/dsf),int(im.shape[0]/dsf)),interpolation = cv2.INTER_AREA)
#             else:
#                 im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 gray = np.empty([total_frames,int(im.shape[0]/dsf),int(im.shape[1]/dsf)])
#                 gray[i,:,:] = cv2.resize(im,(int(im.shape[1]/dsf),int(im.shape[0]/dsf)),interpolation = cv2.INTER_AREA)
#             pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
#             #print(str(pos_frame)+" frames")
#         else:
#             # The next frame is not ready, so we try to read it again
#             cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
#             print("frame is not ready")
#             # It is better to wait for a while for the next frame to be ready
#             cv2.waitKey(1000)
    
#         if cv2.waitKey(10) == 27:
#             break
#         if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
#             # If the number of captured frames is equal to the total number of frames,
#             # we stop
#             break
#     return gray


import matplotlib.pyplot as plt
import matplotlib as mpl
def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r

def save_obj(root, name,obj ):
    with open(root+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(root,name ):
    with open(root + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
from skimage import draw
def polygon2mask(image_shape, polygon):
    """Compute a mask from polygon.
    Parameters
    ----------
    image_shape : tuple of size 2.
        The shape of the mask.
    polygon : array_like.
        The polygon coordinates of shape (N, 2) where N is
        the number of points.
    Returns
    -------
    mask : 2-D ndarray of type 'bool'.
        The mask that corresponds to the input polygon.
    Notes
    -----
    This function does not do any border checking, so that all
    the vertices need to be within the given shape.
    Examples
    --------
    >>> image_shape = (128, 128)
    >>> polygon = np.array([[60, 100], [100, 40], [40, 40]])
    >>> mask = polygon2mask(image_shape, polygon)
    >>> mask.shape
    (128, 128)
    """
    polygon = np.asarray(polygon)
    vertex_row_coords, vertex_col_coords = polygon.T
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, image_shape)
    mask = np.zeros(image_shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


