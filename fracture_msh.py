## @package fracture_msh
# FEM pre-processing for classical_fracture
# used package:
# numpy
import numpy as np
## generating mesh file for fem computation of structure with fracture.
# for example you can find rectangle.msh under msh directory
class FractureMesh:
    vertex_list=[]
    rectangle_list=[]
    line_list=[]
    line_list_with_sigma=[]
    line_list_with_x_sigma=[]
    def __init__(self,h, w, a, h1, w1):
        self.half_height=h
        self.half_width=w
        self.half_fracture_width=a#represent in vector coordinate form    
        self.grid_height=h1
        self.grid_width=w1    
    def mesh(self):
        length_count = int(2 * self.half_width/self.grid_width)
        height_count = int(2 * self.half_height/self.grid_height)
        fracture_count = int(2 * self.half_fracture_width/self.grid_width)
        point_count = (length_count + 1)*(height_count + 1) 
        #print(point_count)
        FractureMesh.vertex_list = []
        for j in range(0, height_count+1):
            for i in range(0, length_count+1):
                FractureMesh.vertex_list.append([self.grid_width*i, self.grid_height*j])
        for i in range(int((length_count-fracture_count)/2 + 1) , int((length_count+fracture_count)/2 )):
            FractureMesh.vertex_list.append([self.grid_width*i, self.grid_height*(int(height_count/2))])
        FractureMesh.vertex_list = np.asarray(FractureMesh.vertex_list)
        FractureMesh.rectangle_list = []
        for i in range(length_count):
            for j in range(height_count):
                x0 = i+j*(length_count+1)
                FractureMesh.rectangle_list.append([x0, x0+1, x0+length_count+2, x0+length_count+1])
        FractureMesh.rectangle_list = np.asarray(FractureMesh.rectangle_list)
        #for i in range(int((length_count-fracture_count)/2) + 1, int((length_count+fracture_count)/2)):
        for i in range(height_count*int((length_count-fracture_count)/2)+int(height_count/2) +height_count , height_count*int((length_count+fracture_count)/2) ,height_count):
            #print(i, ' rectangle_index')
            FractureMesh.line_list.append([FractureMesh.rectangle_list[i, 0],FractureMesh.rectangle_list[i, 1]])        
            FractureMesh.rectangle_list[i, 0] = int((i - (height_count*int((length_count-fracture_count)/2)+1 +height_count))/height_count )+ point_count 
        FractureMesh.line_list.append([FractureMesh.line_list[0][0]-1,FractureMesh.line_list[0][0]])
        for i in range(height_count*int((length_count-fracture_count)/2)+int(height_count/2)  , height_count*int((length_count+fracture_count)/2)-height_count ,height_count):
            #print(i, ' rectangle_index')
            FractureMesh.rectangle_list[i, 1] = int((i -  (height_count*int((length_count-fracture_count)/2)+1))/height_count) + point_count 
            FractureMesh.line_list.append([FractureMesh.rectangle_list[i, 0],FractureMesh.rectangle_list[i, 1]])
        i+=height_count
        FractureMesh.line_list.append([FractureMesh.rectangle_list[i, 0],FractureMesh.rectangle_list[i, 1]])
        for i in range(length_count):
            FractureMesh.line_list_with_sigma.append([i,i+1])
            FractureMesh.line_list_with_sigma.append([i+height_count*(length_count+1),i+height_count*(length_count+1)+1])
        vertex_id=0
        for i in range(height_count):
            FractureMesh.line_list_with_x_sigma.append([vertex_id,vertex_id+length_count+1])
            vertex_id+=length_count+1
        vertex_id=length_count
        for i in range(height_count):
            FractureMesh.line_list_with_x_sigma.append([vertex_id,vertex_id+length_count+1])
            vertex_id+=length_count+1
        
    def write_msh(self,fileName):#transform vertex coordinate to cartasian
        f=open(fileName,'w')
        f.write('$MeshFormat\n\
2.2 0 8\n\
$EndMeshFormat\n\
$Nodes\n')
        vertex_num=len(FractureMesh.vertex_list)
        f.write(str(vertex_num))
        f.write('\n')
        #interpolate between inner_radius and outer_radius
        index=1
        for point in FractureMesh.vertex_list:#write vertex
            f.write('{0:d} {1:.5f} {2:5f} {3:5f}\n'.format(index,point[0],point[1],0))
            index+=1
        f.write('$EndNodes\n\
$Elements\n')
        f.write(str(len(FractureMesh.rectangle_list)+len(FractureMesh.line_list)+len(FractureMesh.line_list_with_sigma)+len(FractureMesh.line_list_with_x_sigma))+'\n')
        index=1
        for element in FractureMesh.rectangle_list:
            f.write('{0:d} 3 2 1 1 {1:d} {2:d} {3:d} {4:d}\n'.format(index,element[0]+1,element[1]+1,element[2]+1,element[3]+1))#write surface quad on inner surface
            index+=1
        for element in FractureMesh.line_list:
            f.write('{0:d} 1 2 0 1 {1:d} {2:d}\n'.format(index,element[0]+1,element[1]+1))
            index+=1
        for element in FractureMesh.line_list_with_sigma:
            f.write('{0:d} 1 2 1 1 {1:d} {2:d}\n'.format(index,element[0]+1,element[1]+1))
            index+=1
        for element in FractureMesh.line_list_with_x_sigma:
            f.write('{0:d} 1 2 2 1 {1:d} {2:d}\n'.format(index,element[0]+1,element[1]+1))
            index+=1
            
        f.write('$EndElements')
        f.close()
if __name__ == '__main__':
    #h, w, a, h1, w1
    #real data
    #2,1,0.2,0.1,0.05
    #test data
    #4,6,3,1,1
    #s=FractureMesh(h=4,w=6,a=3,h1=1,w1=1)
    s=FractureMesh(h=2,w=1,a=0.2,h1=0.1,w1=0.05)
    s.mesh()
    s.write_msh('rectangle.msh')
