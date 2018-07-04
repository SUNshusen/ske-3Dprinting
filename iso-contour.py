##################################################
# This script provides an example for 
# 1. extract contours from mask image [slice]
# 2. generate iso-contours 
# Todo:
# decomposite domain 
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
def generate_iso_contour(contour, offset):
    inter_contour = gen_internal_contour_by_clipper(contour, offset) 
    N = len(inter_contour)
    if N != 0:
        for c in inter_contour:
            cc = np.array(c)
            draw_line(np.vstack([cc, cc[0]]), img, [0,255,0])
            generate_iso_contour(cc, offset)

if __name__ == '__main__':
    offset = -20 # inner offset
    nSamle = 1000 # number of resample vertices
    
    im, contours, areas, hiearchy, root_contour_idx = generate_contours_from_img("C:/Users/hero/Desktop/iso-contour/two-circle.png")
    height, width = im.shape[0], im.shape[1] # picture's size
    img = np.zeros((height, width, 3), np.uint8) + 255 # for demo
    
    vc = get_valid_contours(contours, areas, hiearchy, root_contour_idx) # remove contours that area < 5 (estimated value)
    color_list = generate_RGB_list(int(200/np.abs(offset))) # for demo
    cv2.drawContours(img, vc, -1, (0,0,255), 2) # draw original contour on slice surface
    
    solution = [] # input contours: include outer shape contours and inner hole contours
    for idx in range(0, len(vc)):
        c = np.reshape(vc[idx], (vc[idx].shape[0],2))
        #c = resample_list(c, nSamle)
        solution.append(c)
     
    nTimes = 0 # for demo colors
    while len(solution) != 0: # in order to 
        solution = gen_internal_contour_by_clipper(solution, offset)
        for c in solution:
            cc = np.array(c)
           #draw_line(np.vstack([cc, cc[0]]), img, color_list[nTimes])
            draw_line(cc, img, color_list[nTimes])
        nTimes += 1
    cv2.imshow("Art", img)
    cv2.waitKey(0)

