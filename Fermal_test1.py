##################################################
# This script provides an example for 
# 1. extract contours from mask image [simulate stl slicing]
# 2. generate iso-contours [generate filling path  
# 3. connect iso-contour to fermal spirals
# Todo:
# 1. smooth curve 
# 2. decomposite filling domain 
import cv2
import numpy as np

####################################
# draw poly line to image #
# point_list is n*2 np.ndarray #
####################################
def draw_line(point_lists, img, color, line_width=1):
    point_lists = point_lists.astype(int)
    pts = point_lists.reshape((-1,1,2))
    cv2.polylines(img, [pts], False, color, thickness=line_width, lineType=cv2.LINE_AA)
    cv2.imshow("Art", img)

#############################################
# Generate iso_contour from by clipper #
# https://github.com/greginvm/pyclipper #
# number of input contour: 0-n #
#############################################
import pyclipper
def gen_internal_contour_by_clipper(contours, offset):
    pco = pyclipper.PyclipperOffset()
    pco.AddPaths(contours, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(offset)
    return solution

#######################################################################################
# Generate hiearchy contours from image #
# return contours(python list by a tree) #
# https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
# This function provides:
# 1. Compute area for each contour
# 2. Return image, contours, areas, hiearchy, root_contour_idx 
#######################################################################################
def generate_contours_from_img(imagePath, threath_area = 5):
    verts = []
    im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(im, 127, 255, 1)
    
    image, contours, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    # find contour with largest area
    idx = 0
    selectIdx = -1;
    area = 0
    areas = []
    if len(contours) > 0:
        for c in contours:
            a = cv2.contourArea(c)
            areas.append(a)
            if a > area:
                selectIdx = idx;
        verts = contours[selectIdx]
    return im, contours, areas, hiearchy, selectIdx  

#######################################
# Remove small contours 
#######################################
def get_valid_contours(contours, areas, hiearchy, root_contour_index):
    vc = []
    for i in range(0, len(contours) ):
        if(areas[i] > 5):
            vc.append(contours[i])
            
    return vc
####################################
# Resample List (N < input_size/2) #
####################################
def resample_list(input_list, N):
    input_size = input_list.shape[0]
    N = N if N < input_size/2 else int(input_size/2)
    if N > input_size: return input_list
    Sample = np.linspace(0, input_size, N, dtype = int, endpoint=False)   
    out_list = input_list[Sample]
    return out_list   

#######################################
# Generate N color list
#################################
def generate_RGB_list(N):
    import colorsys
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    rgb_list = tuple(RGB_tuples)
    return np.array(rgb_list) * 255


######################################
# Resample List (N < input_size/2) #
# input_list: contour in m*2 ndarray #
######################################
def resample_list(input_list, N):
    input_size = input_list.shape[0]
    N = N if N < input_size/2 else int(input_size/2)
    Sample = np.linspace(0, input_size, N, dtype = int, endpoint=False)   
    out_list = input_list[Sample]
    return out_list        
#############################################
# Recursively generate iso contour #
#############################################
def generate_iso_contour(contour, offset, is_draw = False):
    global gContours
    inter_contour = gen_internal_contour_by_clipper(contour, offset) 
    N = len(inter_contour)
    if N != 0:
        for c in inter_contour:
            cc = np.array(c)
            if(is_draw):
                draw_line(np.vstack([cc, cc[0]]), img, [0,255,0])
            gContours.append(cc)
        
        generate_iso_contour(inter_contour, offset)

def prev_idx(idx, contour):    
    if idx > 0:
        return idx - 1;
    if idx == 0:
        return len(contour) -1

def next_idx(idx, contour):
    if idx < len(contour)-1:
        return idx + 1
    if idx == len(contour) - 1:
        return 0
#######################################
# find nearest index of point in a contour
# @in: point value(not index), a contour
# @out: the index of a nearest point
#####################################
def find_nearest_point_idx(point, contour):
    idx = -1
    distance = float("inf")   
    for i in range(0,contour.shape[0]-1):        
        d = np.linalg.norm(point-contour[i])
        if d < distance:
            distance = d
            idx = i
    return idx
##########################################################################
#find an index of point from end, with distance larger than T
#@in: current index of point, current contour, 
#@in: T is a distance can be set to offset or offset/2 or 2 * offset
##########################################################################
def find_point_index_by_distance(cur_point_index, cur_contour, T):
    T = abs(T)
    start_point = cur_contour[cur_point_index]
    idx_end_point = prev_idx(cur_point_index, cur_contour)
    
    end_point=[]        
    for ii in range(0,len(cur_contour)-1):
        end_point = cur_contour[idx_end_point]
        distance=np.linalg.norm(start_point-end_point)            
        if distance > T:           
            break
        else:         
            print(idx_end_point)
            idx_end_point = prev_idx(idx_end_point, cur_contour)  
    return idx_end_point

##############################################################
# @in: iso contours, index of start point, offset
# @out: a single poly
# If you want connect in then connect out,
# you can divide contours into two sets, and run it twice,
# then connect them.
##############################################################
def contour2spiral(contours, idx_start_point, offset):
    offset = abs(offset)
    cc = [] # contour for return
    N = len(contours)
    for i in range(N):
        contour1 = contours[i]        
        
        ## find end point(e1)
        idx_end_point = find_point_index_by_distance(idx_start_point, contour1, 2*offset)
        end_point = contour1[idx_end_point]
        
        # add contour segment to cc
        idx = idx_start_point
        while idx != idx_end_point:
            cc.append(contour1[idx])
            idx = next_idx(idx, contour1)   
        
        if(i == N-1): 
            break
        
        ## find s2   
        idx_start_point2 = find_nearest_point_idx(end_point, contours[i+1])         
        
        idx_start_point = idx_start_point2   
        
        
    return cc     

def connect_spiral(first_spiral, second_spiral, is_flip=True):
    s = []
    if is_flip:
        second_spiral = np.flip(second_spiral, 0)
        
    for i in range(len(first_spiral)):
        s.append(first_spiral[i])                 
    for i in range(len(second_spiral)):
        s.append(second_spiral[i])
    return s

from scipy.signal import savgol_filter
def smooth_curve_by_savgol(c, filter_width=5, polynomial_order=1):
    N = 10
    c = np.array(c)
    y = c[:, 1]
    x = c[:, 0]
    x2 = savgol_filter(x, filter_width, polynomial_order)
    y2 = savgol_filter(y, filter_width, polynomial_order)
    c = np.transpose([x2,y2])
    return c


if __name__ == '__main__':
    offset = -11 # inner offset
    nSamle = 1000 # number of resample vertices
    gContours = []
    
    im, contours, areas, hiearchy, root_contour_idx = generate_contours_from_img("./data/circle.png")
    height, width = im.shape[0], im.shape[1]             # picture's size
    img = np.zeros((height, width, 3), np.uint8) + 255   # for demostration
    
    vc = get_valid_contours(contours, areas, hiearchy, root_contour_idx) # remove contours that area < 5 (estimated value)
    color_list = generate_RGB_list(int(200/np.abs(offset))) # for demo    
    
    solution = [] # input contours: include outer shape contours and inner hole contours
    for idx in range(0, len(vc)):
        c = np.reshape(vc[idx], (vc[idx].shape[0],2))
        #c = resample_list(c, len(c)/1)
        c = np.flip(c,0)    # reverse index order
        solution.append(c)    
        
    gContours.append(solution[0])
    generate_iso_contour(solution, offset)
    
    #connect
    ## divide contours into two groups(by odd/even)
    in_contour_groups = []
    out_contour_groups = []
    for idx in range(len(gContours)):
        if (idx % 2 == 0):
            in_contour_groups.append(gContours[idx])
        else:
            out_contour_groups.append(gContours[idx])
            
    
    cc_in = contour2spiral(in_contour_groups, 0, offset )
    output_index = find_nearest_point_idx(in_contour_groups[0][0], out_contour_groups[0]) 
           
    cc_out = contour2spiral(out_contour_groups, output_index, offset )
    
    ## connect two spiral
    fspiral = connect_spiral(cc_in, cc_out)
    ## set out point
    out_point_index = find_point_index_by_distance(0, in_contour_groups[0], offset)
    fspiral.append(in_contour_groups[0][out_point_index])   
    ## smooth withe filter size 7, order 1
    fspiral = smooth_curve_by_savgol(fspiral, 5, 1)
    draw_line(np.array(fspiral), img, [255, 0, 0], 1) 
    
    #draw point
    cv2.circle(img,tuple(in_contour_groups[0][0]), 4, (0, 0, 255), -1)
    cv2.circle(img,tuple(in_contour_groups[0][out_point_index]), 4, (0, 0, 255), -1)
  
    cv2.imshow("Art", img)
    cv2.waitKey(0)    