## @package traction_rod
# Data pre-processing for traction_rod.
# draw a closed polygonal line and extrude it.
# output format: .geo, which is input format of gmsh
# cmd: gmsh.exe traction_rod.geo, and .msh mesh file can be get
#
# **traction_rod1.geo** -> **traction_rod1.msh**, which is a simple version of traction rod
#
# **traction_rod2.geo**-> **traction_rod2.msh**, which is a middle version of traction rod
#
# **traction_rod3.geo** -> **traction_rod3.msh**, which is a complex version of traction rod
#
# **tration_rod3.geo** simply use **traction_rod.stl** as input file and reclassify it to remesh.
#
# **traction_rod.stl** is produced by parsing the original traction rod stl entity model
#
# all these files can be found under the msh directory.
#
# under the same directory,**traction_rod_model.blend** is blender database file, which is a simple traction rod model
# I made. 
from math import *
if __name__ == '__main__':
    R=0.77;
    alpha=0.48;
    K1=0.95;
    beta=1.12;
    K2=1.0;
    N=5;#circle split
    Ls1=[]
    Ls2=[]
    F_total=2;
    rodHeight=0.7;
    alpha_tmp=(pi-alpha)/N;
    R*2*sin(alpha_tmp/2)
    normal_force_per_face=F_total/(2*sin((pi-alpha_tmp)/2)*rodHeight*R*2*sin(alpha_tmp/2))
    print("normal force per face:%f"%normal_force_per_face)
    for i in range(N):
        Ls1.append(R*cos(pi/2-i*(pi-alpha)/N))
        Ls2.append(R*sin(pi/2-i*(pi-alpha)/N)+R*cos(alpha)+K1*sin(beta)+K2)
    Ls1.append(Ls1[-1]-K1*cos(beta))
    Ls2.append(Ls2[-1]-K1*sin(beta))
    Ls1.append(Ls1[-1])
    Ls2.append(Ls2[-1]-K2)
    for i in range(1,N+2):
        Ls1.append(Ls1[N+1-i])
        Ls2.append(-Ls2[N+1-i])
    for i in range(1,2*(N+1)):
        Ls1.append(-Ls1[2*N+2-i])
        Ls2.append(Ls2[2*N+2-i])
    st=""
    for i in range(len(Ls1)):
        st+="Point(%d)={%f,%f,0};\n"%(i+1,Ls1[i],Ls2[i])
    for i in range(len(Ls1)):
        st+="Line(%d)={%d,%d};\n"%(i+1,i+1,(i+1)%len(Ls1)+1)
    print(st)
#line loop here
#pyplot.plot(Ls1,Ls2)
#pyplot.show()
