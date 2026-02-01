import math as mt

###########################
#### define the generic information
N = 80.0
a = 5e-5  # the first layer height
r1 = 1.2
hn1 =20   # the number of the grids in the boundary layer 15
nozzle_width = 0.04
length = 10*nozzle_width
H = 0.04*6
###########################

q = mt.pow(r1, hn1)
L1 = a*(1-mt.pow(r1, hn1))/(1-r1)

with open('paras2.H', 'w') as f:
    f.write('SV (');
    f.write('('+repr(L1/H)+' '+repr(hn1/N)+' '+repr(q)+')\n');
    f.write('('+repr(0.5)+' '+repr(0.3)+' '+repr(4)+')\n');
    f.write('('+repr(1-L1/H-0.5)+' '+repr(1-hn1/N-0.3)+' '+repr(0.02)+')\n);');
    
f.close()
   
   
   
   
   
   
   
   
   
   
   

