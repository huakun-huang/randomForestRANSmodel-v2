# randomForestRANSmodel-v2
In our previous work, we presented a machine learning model for complex flows. <br>


Huakun Huang, Qingmo Xie, Tai'an Hu, Huan Hu, Peng Yu, A random forest machine learning in turbulence closure modeling for complex flows and heat transfer based on the non-equilibrium turbulence assumption, Journal of Computational Physics, 2025, 533, 113995, doi: 10.1016/j.jcp.2025.113995 <br>


Now, The present study aims to address challenging problems concerning 
the representation of 3D instabilities within 2D simulations. 
Therefroe, this update includes the important features for the above purpose.


<h2><strong>Cases used in the training process</strong></h2>
the previous training set is continually used as the training data. These old training cases are the 3D swirling pipe flows (SW), jet impingement (JIMP), zero-pressure-gradient flat plate flows (T3A/B), and backward-facing step (PitzDaily). It is worth stressing that for round jet impingement flows (Series for H/D_j cases, where H is the impinging distance and D_j is the nozzle diameter), they are the 2D axisymmetric. Therefore, the vortex stretching can occur, while this flow physics vanishes in the plane jet impingement (Series for H/B cases). <br>


**New features**: To enhancing the performance of the SST-ML model in predicting the complex flow physics related to the flow past a cylinder, we newly add the additional jet impingement flows, the flows past the cylinder (FPC), the high-speed flows over the NACA0012 airfoil (FPNA), channel flow, and the jet flow in a Laval nozzle (JF).

| 左对齐 | 居中对齐 | 右对齐 |
| :--- | :---: | ---: |
| 内容1 | 内容2 | 内容3 |
| 内容4 | **加粗** | `代码` |
