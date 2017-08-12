Merge "traction_rod.stl";
RefineMesh;

// Create the topology of the discrete model
CreateTopology;

// We can now define a compound line (resp. surface) for each discrete line
// (resp. surface) in the model
ll[] = Line "*";
For j In {0 : #ll[]-1}
  Compound Line(newl) = ll[j];
EndFor
ss[] = Surface "*";
s = news;
For i In {0 : #ss[]-1}
  Compound Surface(s+i) = ss[i];
  Printf("%g",s+i);
EndFor

// And we can create the volume based on the new compound entities
Surface Loop(1) = {s : s + #ss[]-1};
Volume(1) = {1};
Mesh.CharacteristicLengthMax = 0.05;
