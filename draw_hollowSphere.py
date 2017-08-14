## @package draw_hollowSphere
# Data pre-processing for hollowSphereEighth.
# draw 1/8 hollowSphere
from math import *
import sys

## class for accomplishing the task of generating mesh.
class SphereMesh:
    ## triangle index list
    perm3=[[0,1],[1,2],[2,0]]
    ## triangle cell refinement index list
    perm4=[[0,3,5],[1,3,4],[4,5,2],[3,4,5]]
    ## rectangle cell index list (mode I)
    perm5=[[0,5,6,3],[3,6,4,1],[6,5,2,4]]
    ## rectangle cell index list (mode II)    
    perm5_2=[[0,3,6,5],[3,1,4,6],[6,4,2,5]]
    ## The constructor.
    # @param inner_radius hollow sphere inner radius
    # @param outer_radius hollow sphere outer radius
    def __init__(self,inner_radius=1.0,outer_radius=1.0):
        self.inner_radius=inner_radius
        self.outer_radius=outer_radius
        self.triangle_list=[[(0,1,2)]]#represent in vector coordinate form    
        self.vertex_list=[[0,0],[1,0],[0,1]]
        self.isRecombined=False
    ## implementing the refinement.
    #  @param self The object pointer.
    #  @param n refinement level  
    def refine(self,n):
        if(self.isRecombined):
            raise Exception('Sphere Mesh is already recombined!')
        for j in range(n):
            next_triangle_collection=[]
            current_triangle_collection=self.triangle_list[j]
            for triangle in current_triangle_collection:
                tmp_triangle_vertex_list=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
                for k in range(3):#triangle has 3 vertices
                    tmp_triangle_vertex_list[k][0]=self.vertex_list[triangle[k]][0]
                    tmp_triangle_vertex_list[k][1]=self.vertex_list[triangle[k]][1]
                for k in range(3):#get the three mid-point
                    tmp_triangle_vertex_list[k+3][0]=(tmp_triangle_vertex_list[self.perm3[k][0]][0]+tmp_triangle_vertex_list[self.perm3[k][1]][0])/2.0
                    tmp_triangle_vertex_list[k+3][1]=(tmp_triangle_vertex_list[self.perm3[k][0]][1]+tmp_triangle_vertex_list[self.perm3[k][1]][1])/2.0
                    if (self.vertex_list.count(tmp_triangle_vertex_list[k+3])==0):
                        self.vertex_list.append(tmp_triangle_vertex_list[k+3])
                for k in range(4):
                    next_triangle_collection.append((self.vertex_list.index(tmp_triangle_vertex_list[self.perm4[k][0]]),
                                                    self.vertex_list.index(tmp_triangle_vertex_list[self.perm4[k][1]]),
                                                    self.vertex_list.index(tmp_triangle_vertex_list[self.perm4[k][2]])))
            self.triangle_list.append(next_triangle_collection)
    ## recombine to generate three rectangles per triangle.
    def recombine(self):
        next_triangle_collection=[]
        current_triangle_collection=self.triangle_list[len(self.triangle_list)-1]
        for triangle in current_triangle_collection:
            tmp_triangle_vertex_list=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
            for k in range(3):#triangle has 3 vertices
                tmp_triangle_vertex_list[k][0]=self.vertex_list[triangle[k]][0]
                tmp_triangle_vertex_list[k][1]=self.vertex_list[triangle[k]][1]
            for k in range(3):#get the three mid-point
                tmp_triangle_vertex_list[k+3][0]=(tmp_triangle_vertex_list[self.perm3[k][0]][0]+tmp_triangle_vertex_list[self.perm3[k][1]][0])/2.0
                tmp_triangle_vertex_list[k+3][1]=(tmp_triangle_vertex_list[self.perm3[k][0]][1]+tmp_triangle_vertex_list[self.perm3[k][1]][1])/2.0
                if (self.vertex_list.count(tmp_triangle_vertex_list[k+3])==0):
                    self.vertex_list.append(tmp_triangle_vertex_list[k+3])
            tmp_triangle_vertex_list[6][0]=(tmp_triangle_vertex_list[0][0]+tmp_triangle_vertex_list[1][0]+tmp_triangle_vertex_list[2][0])/3#center_x
            tmp_triangle_vertex_list[6][1]=(tmp_triangle_vertex_list[0][1]+tmp_triangle_vertex_list[1][1]+tmp_triangle_vertex_list[2][1])/3#center_y  
            self.vertex_list.append(tmp_triangle_vertex_list[6])
            for k in range(3):
                x1=tmp_triangle_vertex_list[1][0]-tmp_triangle_vertex_list[0][0]
                y1=tmp_triangle_vertex_list[1][1]-tmp_triangle_vertex_list[0][1]
                x2=tmp_triangle_vertex_list[2][0]-tmp_triangle_vertex_list[0][0]
                y2=tmp_triangle_vertex_list[2][1]-tmp_triangle_vertex_list[0][1]
                if(x1*y2-x2*y1>0):
                    next_triangle_collection.append((self.vertex_list.index(tmp_triangle_vertex_list[self.perm5[k][0]]),
                                                    self.vertex_list.index(tmp_triangle_vertex_list[self.perm5[k][1]]),
                                                    self.vertex_list.index(tmp_triangle_vertex_list[self.perm5[k][2]]),
                                                    self.vertex_list.index(tmp_triangle_vertex_list[self.perm5[k][3]])                                                
                                                    ))
                else:
                    next_triangle_collection.append((self.vertex_list.index(tmp_triangle_vertex_list[self.perm5_2[k][0]]),
                                                    self.vertex_list.index(tmp_triangle_vertex_list[self.perm5_2[k][1]]),
                                                    self.vertex_list.index(tmp_triangle_vertex_list[self.perm5_2[k][2]]),
                                                    self.vertex_list.index(tmp_triangle_vertex_list[self.perm5_2[k][3]])                                                
                                                    ))
                
        self.triangle_list.append(next_triangle_collection)        
        self.isRecombined=True
    ## implementing mapping to sphere surface and write the output in msh format    
    def write_msh(self,fileName,Layer_Num=1):#transform vertex coordinate to cartasian
        f=open(fileName,'w')
        f.write('$MeshFormat\n\
2.2 0 8\n\
$EndMeshFormat\n\
$Nodes\n')
        if(not self.isRecombined and Layer_Num>1):
            raise Exception('Not Recombined Triangles do not allow more than one layer')
        vertex_num=Layer_Num*len(self.vertex_list)
        f.write(str(vertex_num))
        f.write('\n')
        #interpolate between inner_radius and outer_radius
        if(Layer_Num>1):
            increment=(self.outer_radius-self.inner_radius)*1.0/(Layer_Num-1)
        else:
            increment=0
        index=1
        for j in range(Layer_Num):
            r=self.inner_radius+increment*j
            for point in self.vertex_list:#write vertex
                if(self.isRecombined):
                    x1=1-point[0]-point[1]
                    y1=point[0]
                    z1=point[1]
                    r1=sqrt(x1**2+y1**2+z1**2)
                    x=r*x1/r1
                    y=r*y1/r1
                    z=r*z1/r1
                else:
                    x=r*(point[0]+point[1]*cos(pi/3))
                    y=r*point[1]*sin(pi/3)
                    z=0
                f.write('{0:d} {1:.5f} {2:5f} {3:5f}\n'.format(index,x,y,z))
                index+=1
        f.write('$EndNodes\n\
$Elements\n')
        current_refinement_triangle_list=self.triangle_list[len(self.triangle_list)-1]
        hexagonal_num=(Layer_Num-1)*len(current_refinement_triangle_list)
        neumann_boundary_surface_num=2*len(current_refinement_triangle_list)#only consider inner pressure here
        dirichlet_boundary_surface_num=(Layer_Num-1)*3*int(pow(2,len(self.triangle_list)-1))
        f.write(str(neumann_boundary_surface_num+hexagonal_num+dirichlet_boundary_surface_num)+'\n')
        print('Statistics: hexagonal_num:%s\nneumann_boundary_surface_num:%s\ndirichlet_boundary_surface_num:%s'%(hexagonal_num,neumann_boundary_surface_num,dirichlet_boundary_surface_num))
        index=1
        v=len(self.vertex_list)
        for element in current_refinement_triangle_list:
            if(self.isRecombined):#detect boundary here
                atBoundary=False
                for i in range(4):
                    point=self.vertex_list[element[i]]
                    if(abs(point[0])<1e-5 or abs(point[1])<1e-5 or abs(point[0]+point[1]-1)<1e-5):
                        atBoundary=True
                        break
                if(atBoundary and Layer_Num>1):#at which boundary? x,y,z                    
                    atXBoundary=0
                    elementX=[0,0]
                    atYBoundary=0
                    elementY=[0,0]
                    atZBoundary=0
                    elementZ=[0,0]
                    for i in range(4):
                        point=self.vertex_list[element[i]]
                        if(abs(point[0])<1e-5):
                            elementY[atYBoundary]=element[i]
                            atYBoundary+=1
                        if(abs(point[1])<1e-5):
                            elementZ[atZBoundary]=element[i]
                            atZBoundary+=1
                        if(abs(point[0]+point[1]-1)<1e-5):
                            elementX[atXBoundary]=element[i]                        
                            atXBoundary+=1
                    for j in range(Layer_Num-1):
                        if(atXBoundary==2):
                            f.write('{0:d} 3 2 1 1 {1:d} {2:d} {3:d} {4:d}\n'.format(index,elementX[0]+1+j*v,elementX[1]+1+j*v,elementX[1]+1+(j+1)*v,elementX[0]+1+(j+1)*v))#write surface quad on inner surface
                            index+=1
                        if(atYBoundary==2):
                            f.write('{0:d} 3 2 2 1 {1:d} {2:d} {3:d} {4:d}\n'.format(index,elementY[0]+1+j*v,elementY[1]+1+j*v,elementY[1]+1+(j+1)*v,elementY[0]+1+(j+1)*v))#write surface quad on inner surface
                            index+=1
                        if(atZBoundary==2):
                            f.write('{0:d} 3 2 3 1 {1:d} {2:d} {3:d} {4:d}\n'.format(index,elementZ[0]+1+j*v,elementZ[1]+1+j*v,elementZ[1]+1+(j+1)*v,elementZ[0]+1+(j+1)*v))#write surface quad on inner surface
                            index+=1
                f.write('{0:d} 3 2 0 1 {1:d} {2:d} {3:d} {4:d}\n'.format(index,element[0]+1,element[1]+1,element[2]+1,element[3]+1))#write surface quad on inner surface
                index+=1
                f.write('{0:d} 3 2 0 1 {1:d} {2:d} {3:d} {4:d}\n'.format(index,element[0]+1+v*(Layer_Num-1),element[1]+1+v*(Layer_Num-1),element[2]+1+v*(Layer_Num-1),element[3]+1+v*(Layer_Num-1)))#write surface quad on outer surface
                
            else:
                f.write('{0:d} 2 2 0 1 {1:d} {2:d} {3:d}\n'.format(index,element[0]+1,element[1]+1,element[2]+1))
                
            index+=1
            for j in range(Layer_Num-1):
                f.write('{0:d} 5 2 1 1 {1:d} {2:d} {3:d} {4:d} {5:d} {6:d} {7:d} {8:d}\n'.format(index,element[0]+1+j*v,element[1]+1+j*v,element[2]+1+j*v,element[3]+1+j*v,
                element[0]+1+(j+1)*v,element[1]+1+(j+1)*v,element[2]+1+(j+1)*v,element[3]+1+(j+1)*v))#write hexagonal
                index+=1
        f.write('$EndElements')
        f.close()
    ## @var triangle_list
    #  list of triangles
    
    ## @var vertex_list
    # list of vertices
    
    ## @var isRecombined
    # a boolean value to indicate whether the class has been recombined to rectangle state.Only non-recombined state allows refinment

    
if __name__ == '__main__':    
    if(len(sys.argv)<3):
        print("usage: python draw_hollowSphere.py outputFileNamePrefix refinementLevel")
        exit(0)    
    fileName=sys.argv[1]
    refineMementLevel=int(sys.argv[2])
    s=SphereMesh(inner_radius=1,outer_radius=2)
    s.refine(refineMementLevel)
    s.recombine()
    #print(s.vertex_list)
    #print(s.triangle_list[refineMementLevel+1])
    s.write_msh('msh/'+fileName+'.msh',Layer_Num=refineMementLevel*4)
        
