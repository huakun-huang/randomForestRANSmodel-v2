para0  0.0; //ddt
para1  1.0; //timeSstep
para2  1.0; //Co
para3  0.0; //grad
para4  0.0; //grad_value
para5  0.0; //divU
para6  3.0; //divP
para7  2.0; //divk
para8  2.0; //divOmega
para9  0.0; //divU_value
para10  0.0; //divP_value
para11  0.5; //divk_value
para12  0.5; //divOmega_value
para13  0.2; //Relaxation U
para14  0.5; //Relaxation k
para15  0.5; //Relaation omega
para16  0.5; //Relaxation p
para17  0.0; //add time step
para18  1.0; //cleanCase
para19  1.0; //timeRitio
time 40000; //endTime
ddts  steadyState;
grad Gauss linear;
divU bounded Gauss linearUpwind grad;
divP bounded Gauss upwind;
divk bounded Gauss limitedLinear $para11;
divomega bounded Gauss limitedLinear $para12;
