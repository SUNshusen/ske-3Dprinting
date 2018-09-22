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
    offset = -0.4 # inner offset
    nSamle = 1000 # number of resample vertices
    gContours = []
    
    im, contours, areas, hiearchy, root_contour_idx = generate_contours_from_img("C:/Users/hero/Desktop/iso-contour/circle1.png")
    height, width = im.shape[0], im.shape[1]             # picture's size
    img = np.zeros((height, width, 3), np.uint8) + 255   # for demostration
    
    vc = get_valid_contours(contours, areas, hiearchy, root_contour_idx) # remove contours that area < 5 (estimated value)
    color_list = generate_RGB_list(int(200/np.abs(offset))) # for demo    
    
    solution = [] # input contours: include outer shape contours and inner hole contourscircle
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
    E=0
    print("""
; external perimeters extrusion width = 0.40mm
; perimeters extrusion width = 0.67mm
; infill extrusion width = 0.67mm
; solid infill extrusion width = 0.67mm
; top infill extrusion width = 0.67mm

M107
M190 S40 ; set bed temperature
M104 S190 ; set temperature
G28 ; home all axes
G1 Z5 F5000 ; lift nozzle

M109 S190 ; wait for temperature to be reached
G21 ; set units to millimeters
G90 ; use absolute coordinates
M82 ; use absolute distances for extrusion
G92 E0
G1 E-3.00000 F1800.00000
G92 E0
G1 Z0.350 F4800.000""")
    
    print("G1","X"+str("%0.3f"%fspiral[0][0])+" Y"+str("%0.3f"%fspiral[0][1]) +"F4800")
    print("G1 E3.00000 F1800")
    E=3
    for i in range (0,len(fspiral)-1):
        if i==0:
            print("G1","X"+str("%0.3f"%fspiral[i][0])+" Y"+str("%0.3f"%fspiral[i][1]) + " E" + str("%0.3f"%E))
        else:
            distance=np.linalg.norm(fspiral[i]-fspiral[i-1])
            E=distance*0.0909+E
            print("G1","X"+str("%0.3f"%fspiral[i][0])+" Y"+str("%0.3f"%fspiral[i][1]) + " E" + str("%0.5f"%E))
    print("""G1 E6.95908 F1800.00000
G92 E0
M104 S0 ; turn off temperature
G28 X0  ; home X axis
M84     ; disable motors

; filament used = 148.9mm (0.4cm3)

; avoid_crossing_perimeters = 0
; bed_shape = -100x-100,100x-100,100x100,-100x100
; bed_temperature = 10
; before_layer_gcode = 
; bridge_acceleration = 0
; bridge_fan_speed = 100
; brim_width = 0
; complete_objects = 0
; cooling = 1
; default_acceleration = 0
; disable_fan_first_layers = 1
; duplicate_distance = 6
; end_gcode = M104 S0 ; turn off temperature\nG28 X0  ; home X axis\nM84     ; disable motors\n
; extruder_clearance_height = 20
; extruder_clearance_radius = 20
; extruder_offset = 0x0
; extrusion_axis = E
; extrusion_multiplier = 1,1
; fan_always_on = 0
; fan_below_layer_time = 60
; filament_colour = #FFFFFF
; filament_diameter = 1.75,1.75
; first_layer_acceleration = 0
; first_layer_bed_temperature = 10
; first_layer_extrusion_width = 200%
; first_layer_speed = 30%
; first_layer_temperature = 190,190
; gcode_arcs = 0
; gcode_comments = 0
; gcode_flavor = reprap
; infill_acceleration = 0
; infill_first = 0
; layer_gcode = 
; max_fan_speed = 0
; max_print_speed = 60
; max_volumetric_speed = 0
; min_fan_speed = 35
; min_print_speed = 10
; min_skirt_length = 0
; notes = 
; nozzle_diameter = 0.4
; only_retract_when_crossing_perimeters = 1
; ooze_prevention = 0
; output_filename_format = [input_filename_base].gcode
; perimeter_acceleration = 0
; post_process = 
; pressure_advance = 0
; resolution = 0
; retract_before_travel = 2
; retract_layer_change = 1
; retract_length = 3
; retract_length_toolchange = 10
; retract_lift = 0
; retract_restart_extra = 0
; retract_restart_extra_toolchange = 0
; retract_speed = 30
; skirt_distance = 6
; skirt_height = 1
; skirts = 1
; slowdown_below_layer_time = 30
; spiral_vase = 0
; standby_temperature_delta = -5
; start_gcode = G28 ; home all axes\nG1 Z5 F5000 ; lift nozzle\n
; temperature = 190,190
; threads = 2
; toolchange_gcode = 
; travel_speed = 80
; use_firmware_retraction = 0
; use_relative_e_distances = 0
; use_volumetric_e = 0
; vibration_limit = 0
; wipe = 0
; z_offset = 0
; dont_support_bridges = 1
; extrusion_width = 0
; first_layer_height = 0.35
; infill_only_where_needed = 0
; interface_shells = 0
; layer_height = 0.2
; raft_layers = 0
; seam_position = aligned
; support_material = 0
; support_material_angle = 0
; support_material_contact_distance = 0.2
; support_material_enforce_layers = 0
; support_material_extruder = 1
; support_material_extrusion_width = 0
; support_material_interface_extruder = 1
; support_material_interface_layers = 3
; support_material_interface_spacing = 0
; support_material_interface_speed = 100%
; support_material_pattern = pillars
; support_material_spacing = 2.5
; support_material_speed = 60
; support_material_threshold = 90
; xy_size_compensation = 0
; bottom_solid_layers = 3
; bridge_flow_ratio = 1
; bridge_speed = 60
; external_fill_pattern = rectilinear
; external_perimeter_extrusion_width = 0
; external_perimeter_speed = 70%
; external_perimeters_first = 0
; extra_perimeters = 1
; fill_angle = 45
; fill_density = 20%
; fill_pattern = honeycomb
; gap_fill_speed = 20
; infill_every_layers = 1
; infill_extruder = 1
; infill_extrusion_width = 0
; infill_overlap = 15%
; infill_speed = 60
; overhangs = 1
; perimeter_extruder = 1
; perimeter_extrusion_width = 0
; perimeter_speed = 30
; perimeters = 3
; small_perimeter_speed = 30
; solid_infill_below_area = 70
; solid_infill_every_layers = 0
; solid_infill_extruder = 1
; solid_infill_extrusion_width = 0
; solid_infill_speed = 60
; thin_walls = 1
; top_infill_extrusion_width = 0
; top_solid_infill_speed = 50
; top_solid_layers = 3""")