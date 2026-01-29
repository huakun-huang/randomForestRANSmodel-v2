# randomForestRANSmodel-v2
In our previous work, we presented a machine learning model for complex flows. <br>


Huakun Huang, Qingmo Xie, Tai'an Hu, Huan Hu, Peng Yu, A random forest machine learning in turbulence closure modeling for complex flows and heat transfer based on the non-equilibrium turbulence assumption, Journal of Computational Physics, 2025, 533, 113995, doi: 10.1016/j.jcp.2025.113995 <br>


Now, The present study aims to address challenging problems concerning 
the representation of 3D instabilities within 2D simulations. 
Therefroe, this update includes the important features for the above purpose.


<h2><strong>Cases used in the training process</strong></h2>
the previous training set is continually used as the training data. These old training cases are the 3D swirling pipe flows (SW), jet impingement (JIMP), zero-pressure-gradient flat plate flows (T3A/B), and backward-facing step (PitzDaily). It is worth stressing that for round jet impingement flows (Series for H/Dj cases, where H is the impinging distance and Dj is the nozzle diameter), they are the 2D axisymmetric. Therefore, the vortex stretching can occur, while this flow physics vanishes in the plane jet impingement (Series for H/B cases). <br>


**New features**: To enhancing the performance of the SST-ML model in predicting the complex flow physics related to the flow past a cylinder, we newly add the additional jet impingement flows, the flows past the cylinder (FPC), the high-speed flows over the NACA0012 airfoil (FPNA), channel flow, and the jet flow in a Laval nozzle (JF).

I: Case item; Re: Reynolds number; Dim.: dimension;  Exp.: experiment; Num.: numerical simulation results
| I  | Cases        | Re    | Dim.| Reference model | Vortex stretching | Exp./Num. | Compressible | Heat transfer |
| :--- | :---      | ---:  | ---: | ---:    | ---: | ---: | ---: | ---: |
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


**Geometries** <br>
<img width="683" height="368" alt="Old" src="https://github.com/user-attachments/assets/13f7fa78-0de6-4841-993a-f4049f3bdd5f" />


**Updated cases in this version**:

| I  | Cases        | Re    | Dim.| Reference model | Vortex stretching | Exp./Num. | Compressible | Heat transfer |
| :--- | :---      | ---:  | ---: | ---:    | ---: | ---: | ---: | ---: |
| 15  | FPC         | 100   | 3D  | WALE     | Yes | - | No | No |
| 16  | FPC         | 390   | 2D* | WALE     | Yes | - | No | No |
| 17  | FPC         | 3,900 | 2D  | SSTIDDES | No  | -  | No | No |
| 18  | FPC         | 3,900 | 2D* | WALE     | Yes | -  | No | No |
| 19  | FPC         | 9,000 | 2D* | WALE     | Yes | - | No | No |
| 20  | FPC         | 9,000 | 3D  | WALE     | Yes | - | No | No |
| 21  | JIMP, H/B=2 | 30,000| 2D | SSTCD     | No | - | No | Yes |
| 22  | JIMP, H/B=6 | 11,000| 2D | SSTCD     | No | - | No | Yes |
| 23  | JIMP, H/B=7    | 20,000| 2D | SSTLMCD   | No | - | No | Yes |
| 24  | JIMP, H/Dj=4   | 23,000| 2D | SSTLMCD+VC | Yes | - | No | Yes |
| 25  | JIMP, H/Dj=6   | 70,000| 2D | SSTLMCD+VC | Yes | - | No | Yes |
| 26  | JIMP, H/Dj=1   | 100,000| 2D | SSTLMCD+VC | Yes | - | Yes | Yes |
| 27  | JIMP, H/Dj=5   | 100,000| 2D | SSTLMCD+VC | Yes | - | Yes | Yes |
| 28  | JIMP-C, H/Dj=2 | 23,000 | 2D | SSTLMCD+VC | Yes | - | No | Yes |
| 29  | FPNA, theta=0   | 300,000| 2D | SSTLMCD | No | - | Yes | No |
| 30  | FPNA, theta=10   | 300,000| 2D | SSTLMCD | No | - | Yes | No |
| 31  | JF   | - | 2D | laminar | No | - | Yes | No |
| 32  | Channel flow   | - | 3D | WALE | Yes | - | No | No |


**Geometries** <br>
<img width="952" height="176" alt="ML-geometry" src="https://github.com/user-attachments/assets/5285a58e-0a8e-4548-8264-28f696d36f53" />


<h2><strong>Data portions in different cases</strong></h2>
| data portions | flow type |
|:---:|:---:|
|<img src="https://github.com/user-attachments/assets/1a213b47-e441-4ed0-88d0-0bbb6586f571" width="60%" >| 
<img src="https://github.com/user-attachments/assets/f3918719-9589-42bc-b3c9-2f6517e2d95d" width="60%" > |







