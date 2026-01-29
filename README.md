# randomForestRANSmodel-v2
In our previous work, we presented a machine learning model for complex flows. <br>


Huakun Huang, Qingmo Xie, Tai'an Hu, Huan Hu, Peng Yu, A random forest machine learning in turbulence closure modeling for complex flows and heat transfer based on the non-equilibrium turbulence assumption, Journal of Computational Physics, 2025, 533, 113995, doi: 10.1016/j.jcp.2025.113995 <br>


Now, The present study aims to address challenging problems concerning 
the representation of 3D instabilities within 2D simulations. 
Therefroe, this update includes the important features for the above purpose.


<h2><strong>Cases used in the training process</strong></h2>
the previous training set is continually used as the training data. These old training cases are the 3D swirling pipe flows (SW), jet impingement (JIMP), zero-pressure-gradient flat plate flows (T3A/B), and backward-facing step (PitzDaily). It is worth stressing that for round jet impingement flows (Series for H/Dj cases, where H is the impinging distance and Dj is the nozzle diameter), they are the 2D axisymmetric. Therefore, the vortex stretching can occur, while this flow physics vanishes in the plane jet impingement (Series for H/B cases). <br>


**New features**: To enhancing the performance of the SST-ML model in predicting the complex flow physics related to the flow past a cylinder, we newly add the additional jet impingement flows, the flows past the cylinder (FPC), the high-speed flows over the NACA0012 airfoil (FPNA), channel flow, and the jet flow in a Laval nozzle (JF).

I: Case item; Re: Reynolds number; Dim.: dimension; Ref.: reference model; Exp.: experiment; Num.: numerical simulation results
| I  | Cases        | Re    | Dim.| Reference model | Vortex stretching | Exp./Num. | Compressible | Heat transfer |
| :--- | :---:      | ---:  | ---: | ---:    | ---: | ---: | ---: | ---: |
| 1  | T3A          | 5,281 | 2D | SSTLM     | No | - | No | No |
| 2  | T3A-         | 4,694 | 2D | SSTLM     | No | - | No | No |
| 3  | T3A2-        | 7,726 | 2D | SSTLM     | No | - | No | No |
| 4  | T3B-         | 9,780 | 2D | SSTLM     | No | - | No | No |
| 5  | JIMP, H/B=2  | 11,400| 2D | SSTCD     | No | - | No | Yes |
| 6  | JIMP, H/B=4  | 20,000| 2D | SSTCDLM   | No | - | No | Yes |
| 7  | JIMP, H/B=9.2| 20,000| 2D | SSTCD     | No | - | No | Yes |
| 8  | JIMP, H/Dj=2 | 23,000| 2D | SSTLMCD+VC| Yes | - | No | Yes |
| 9  | JIMP, H/Dj=7 | 23,000| 2D | SSTLMCD+VC| Yes | - | No | Yes |
| 10 | JIMP, H/Dj=10| 23,000| 2D | SSTLMCD+VC| Yes | - | No | Yes |
| 11 | JIMP, H/Dj=14| 23,000| 2D | SSTLMCD+VC| Yes | - | No | Yes |
| 12 | JIMP, H/Dj=1 | 30,000| 2D | SSTLMCD+VC| Yes | - | Yes | Yes |
| 13 | SW           |280,000| 3D | SSTCC     | Yes | - | No | No |
| 14 | PitzDaily    | 25,400| 2D | WALE      | No  | - | No | No |
