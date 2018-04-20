#import bpy  
import math
#from mathutils import Vector  
#we use this code to create a multiscale model.


# weight  
w = 1 
h = 20

 
listOfVectors = []
X = [x / 100.0 for x in range(20, int(2*math.pi * 100), 10)]
R = [r /100 for r in range(2*100,6*100,50)]



i = 0
z = 0

#create inner circles
for r in R:
    for z in range(0, 20):
        if r<4:
         for x in X:
            listOfVectors.append((r * math.cos(x),  r * math.sin(x), z))
            
        else:
             for x in X:
                 cx = r * math.cos(x)
                 cy = r * math.sin(x)
                 if i % 2 == 0:
                      v1 = (cx, cy, 0)
                      v2 = (cx, cy, h)
                 else:
                      v1 = (cx, cy, h)
                      v2 = (cx, cy, 0)
                 i = i+1   
                 listOfVectors.append(v1)
                 listOfVectors.append(v2)
                 


#set model attribute
