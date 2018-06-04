import cv2
import numpy as np
import math
def draw_line(point_lists, img, color):
    line_width = 1
   
    point_lists = point_lists.astype(int)
    pts = point_lists.reshape((-1,1,2))
    cv2.polylines(img, [pts], False, color, thickness=line_width, lineType=cv2.LINE_AA)
    cv2.imshow("Art", img)
def generate_contour(contour_type, rect, sampleN):
    r = min(rect.shape[0:2]) / 3
    verts = []
    if(contour_type == 0):
        step = int(2*math.pi * 100 / sampleN)
        alpha = [x / 100.0 for x in range(0, int(2*math.pi * 100), step)]
       
        verts_x = r * np.sin(alpha)
        verts_y = r * np.cos(alpha)
        verts = np.vstack([verts_x, verts_y]).T
       
        verts = verts + [img.shape[1]/2, img.shape[0]/2]
       
    if(contour_type == 1):
        len = r / 3;
        step = len / (sampleN / 4)
        verts_x1 = np.array(range(0, len, step))
        verts_y1 = np.zeros(verts_x1.shape[0])
    return verts.astype(int)
 
def generate_internal_contour_by_step(contour, step):
    length = contour.shape[0]
   
    # i.e
    # xs = np.array([1,2,3,4,5])
    # xsl = [2,3,4,5,1] #shift left
    # xsr = [5,1,2,3,4] #shift right
    # internal contour
    cl = np.roll(contour, -1, axis=0)
    cr = np.roll(contour, 1, axis=0)
    r1 = cr - contour
    r2 = cl - contour
    r1_norm = np.linalg.norm(r1, 2, 1)
    r2_norm = np.linalg.norm(r2, 2, 1)
    r1 = r1 / r1_norm.reshape((length, 1))
    r2 = r2 / r2_norm.reshape((length, 1))
    laplas = r1 + r2
    laplas_norm = np.linalg.norm(laplas, 2, 1).reshape((length,1))
   
    new_contour = contour + step * (laplas/laplas_norm)
    return new_contour
   
width, height = 640, 480 # picture's size
img = np.zeros((height, width, 3), np.uint8) + 255
img1 = np.zeros((300, 300, 3), np.uint8) + 255
verts = generate_contour(0, img, 15)
length = verts.shape[0]
draw_line(np.vstack([verts, verts[0]]), img, [0,0,255])
iverts= generate_contour(0,img1,15)
draw_line(np.vstack([iverts,iverts[0]]),img,[0,0,255])
inner = verts 
oddinnerstart=[]
oddinnerend=[]
eveninnerstart=[]
eveninnerend=[]
a=inner
c=[]
for i in range(1,6,1):
    max_idx = inner.shape[0] - 1
    inner = generate_internal_contour_by_step(inner,10)
    dist= cv2.pointPolygonTest(iverts,(inner[max_idx][0],inner[max_idx][1]),False)
    if i==2:
        b=inner  
    if i%2!=0:
        a=np.vstack((a,inner))
    elif i>2:
        b=np.vstack((b,inner))
    #for j in range(b.shape[0]-1,0,-1):
        #c.append((b[j][0],b[j][1]))
    
    if dist>0:
        break 
    r1 = inner[0] - inner[max_idx]
    r2 = inner[max_idx] - inner[max_idx - 1]
    if(np.dot(r1,r2) < 0 ):
        inner = np.delete(inner, max_idx, 0)
    #if i>1:
        #if i%2!=0:
            #draw_line(np.vstack([inner, m[i-3]]), img, [255,0,0])
        #else:
            #draw_line(np.vstack([inner, n[i-3]]), img, [255,0,0])
            
   
        
        
#for i in range ()

    #draw_line(np.vstack([inner, inner[0]]), img, [255,0,0])
    #draw_line(a, img, [255,0,0])
    #draw_line(b, img, [255,0,0])
    #draw_line(inner, img, [255,0,0])
    #for j in range(0,inner.shape[0]-1):
        #print("G1 ","X",inner[j][0]-100,"Y",inner[j][1]-10
for j in range (b.shape[0]-1,-1,-1):
    a=np.vstack((a,b[j]))
#draw_line(b, img, [255,0,0])
draw_line(a,img,[255,0,0])
cv2.waitKey(0) # miliseconds, 0 means wait forever

