//set external force apply direction by physical entity id;
n=6;
rodLen=3.5;
radius=0.72;
//unit: mm*10^-2
For i In {1:n}
Point(i) = {radius*Cos(2*i*Pi/n),radius*Sin(2*i*Pi/n)+radius+rodLen/2,0};
Printf("Point %g:%g",i,radius*Sin(2*i*Pi/n)+radius+rodLen/2);
EndFor
myLineLoop[]={};
For i In {1:n}
Line(i) = {i,i%n+1};
myLineLoop[]+={i};
EndFor
Line Loop(1)=myLineLoop[];
Plane Surface(1) = {1};


offset=n;
surfaceIndex=2;
myLineLoop[]={};
For i In {1:n}
Point(offset+i) = {radius*Cos(2*i*Pi/n),radius*Sin(2*i*Pi/n)-radius-rodLen/2,0};
EndFor
myLineLoop={};
For i In {1:n}
Line(offset+i) = {i+offset,i%n+1+offset};
myLineLoop[]+={i+offset};
EndFor
Line Loop(surfaceIndex)=myLineLoop[];
Plane Surface(surfaceIndex) = {surfaceIndex};

Line(offset*2+1)={8,4};
Line(offset*2+2)={5,7};
Line Loop(3)={7,offset*2+1,4,offset*2+2};
Plane Surface(3) = {3};
Recombine Surface(1);
Recombine Surface(2);
Recombine Surface(3);

//BooleanUnion(4) = { Surface{1,2}; Delete; }{ Surface{3}; Delete; };
rodHeight=0.8;
out[]=Extrude {0,0,rodHeight} {
  Surface{1,3}; Layers{4};Recombine;
};
out2[]=Extrude {0,0,rodHeight} {
  Surface{2}; Layers{4};Recombine;
};
Printf("volume number,%g,%g,%g,%g,%g,%g",out[0],out[1],out[2],out2[0],out2[1],out2[5]);
//BooleanUnion(4)={Volume{1,2};Delete;}{Volume{3};Delete;};
Physical Surface(1) = {out[2],out2[5]};
Physical Volume(2)={1,2,3};