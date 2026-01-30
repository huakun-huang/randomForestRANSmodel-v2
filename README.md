# randomForestRANSmodel-v2
In our previous work, we presented a machine learning model for complex flows. <br>


Huakun Huang, Qingmo Xie, Tai'an Hu, Huan Hu, Peng Yu, A random forest machine learning in turbulence closure modeling for complex flows and heat transfer based on the non-equilibrium turbulence assumption, Journal of Computational Physics, 2025, 533, 113995, doi: 10.1016/j.jcp.2025.113995 <br>


Now, the present study aims to address challenging problems concerning 
the representation of 3D instabilities within 2D simulations. 
Therefroe, this update includes the important features for the above purpose.


<h2><strong>(1) Cases used in the training process</strong></h2>
the previous training set is continually used as the training data. These old training cases are the 3D swirling pipe flows (SW), jet impingement (JIMP), zero-pressure-gradient flat plate flows (T3A/B), and backward-facing step (PitzDaily). It is worth stressing that for round jet impingement flows (Series for H/Dj cases, where H is the impinging distance and Dj is the nozzle diameter), they are the 2D axisymmetric. Therefore, the vortex stretching can occur, while this flow physics vanishes in the plane jet impingement (Series for H/B cases). <br>


**New features**: To enhancing the performance of the SST-ML model in predicting the complex flow physics related to the flow past a cylinder, we newly add the additional jet impingement flows, the flows past the cylinder (FPC), the high-speed flows over the NACA0012 airfoil (FPNA), channel flow, and the jet flow in a Laval nozzle (JF).

I: Case item; Re: Reynolds number; Dim.: dimension;  Exp.: experiment; Num.: numerical simulation results <br>
**SSTLM**: Shear stress transport (SST) model with laminar-turbulence transition proposed by [Langtry and Menter (2009)](https://arc.aiaa.org/doi/10.2514/1.42362). <br>
**SSTCD**: SST model with the cross-diffusion correction proposed by [Huang et al. (2019)](https://www.sciencedirect.com/science/article/pii/S0017931018357302). <br>
**SSTLMCD**: SSTLM model with the cross-diffusion correction proposed by [Huang et al. (2025)](https://www.sciencedirect.com/science/article/pii/S1359431124025092) <br>
**VC**: Vortex streching correction, [Huang et al. (2022)](https://www.sciencedirect.com/science/article/pii/S1359431121011339) <br>
**SSTCC**: SST model with curvature correction proposed by [Smirnov and Menter (2009)](https://asmedigitalcollection.asme.org/turbomachinery/article-abstract/131/4/041010/468836/Sensitization-of-the-SST-Turbulence-Model-to?redirectedFrom=fulltext) <br>
WALE: Wall-adapting local eddy-viscosity model<br>

| I  | Cases        | Re    | Dim.| Reference model | Vortex stretching | Exp./Num. | Compressible | Heat transfer |
| :--- | :---      | ---:  | ---: | ---:    | ---: | ---: | ---: | ---: |
| 1  | T3A          | 5,281 | 2D | SSTLM     | No | [Savil](https://www.sciencedirect.com/science/chapter/edited-volume/abs/pii/B9780444898029500599) | No | No |
| 2  | T3A-         | 4,694 | 2D | SSTLM     | No | [Huang et al.](https://www.sciencedirect.com/science/article/pii/S0021999125002785) | No | No |
| 3  | T3A2-        | 7,726 | 2D | SSTLM     | No | [Huang et al.](https://www.sciencedirect.com/science/article/pii/S0021999125002785) | No | No |
| 4  | T3B-         | 9,780 | 2D | SSTLM     | No | [Huang et al.](https://www.sciencedirect.com/science/article/pii/S0021999125002785) | No | No |
| 5  | JIMP, H/B=2  | 11,400| 2D | SSTCD     | No | [Huang et al.](https://www.sciencedirect.com/science/article/pii/S0017931018357302) | No | Yes |
| 6  | JIMP, H/B=4  | 20,000| 2D | [SSTLMCD](https://www.sciencedirect.com/science/article/pii/S0142727X2400198X)   | No | Exp. [1](https://www.sciencedirect.com/science/article/pii/S0894177796001124), [2](https://asmedigitalcollection.asme.org/fluidsengineering/article/123/1/112/459689/Near-Wall-Measurements-for-a-Turbulent-Impinging) | No | Yes |
| 7  | JIMP, H/B=9.2| 20,000| 2D | [SSTCD](https://www.sciencedirect.com/science/article/pii/S0017931018357302)     | No | Exp. [1](https://www.sciencedirect.com/science/article/pii/S0894177796001124), [2](https://asmedigitalcollection.asme.org/fluidsengineering/article/123/1/112/459689/Near-Wall-Measurements-for-a-Turbulent-Impinging) | No | Yes |
| 8  | JIMP, H/Dj=2 | 23,000| 2D | [SSTLMCD+VC](https://www.sciencedirect.com/science/article/pii/S0021999125002785)| Yes | Exp. [1](https://www.sciencedirect.com/science/article/pii/S0017931005802042), [2](https://doi.org/10.1115/1.2911197) | No | Yes |
| 9  | JIMP, H/Dj=7 | 23,000| 2D | [SSTLMCD+VC](https://www.sciencedirect.com/science/article/pii/S0021999125002785)| Yes | -  | No | Yes |
| 10 | JIMP, H/Dj=10| 23,000| 2D | [SSTLMCD+VC](https://www.sciencedirect.com/science/article/pii/S0021999125002785)| Yes | [Baughn et al.](https://doi.org/10.1115/1.2911197) | No | Yes |
| 11 | JIMP, H/Dj=14| 23,000| 2D | [SSTLMCD+VC](https://www.sciencedirect.com/science/article/pii/S0021999125002785)| Yes | [Baughn et al.](https://doi.org/10.1115/1.2911197) | No | Yes |
| 12 | JIMP, H/Dj=1 | 30,000| 2D | [SSTLMCD+VC](https://www.sciencedirect.com/science/article/pii/S0021999125002785)| Yes | - | Yes | Yes |
| 13 | SW           |280,000| 3D | [SSTCC](https://www.sciencedirect.com/science/article/pii/S0017931021010851)     | Yes | - | No | No |
| 14 | PitzDaily    | 25,400| 2D | [WALE](https://www.sciencedirect.com/science/article/pii/S0021999125002785)      | No  | - | No | No |


**Geometries** <br>
<img width="683" height="368" alt="Old" src="https://github.com/user-attachments/assets/13f7fa78-0de6-4841-993a-f4049f3bdd5f" />


**Updated cases in this version**:
**SSTIDDES**: A hybrid RANS/LES method proposed by [Gritskevich et al. (2012)](https://doi.org/10.1007/s10494-011-9378-4)
| I  | Cases        | Re    | Dim.| Reference model | Vortex stretching | Exp./Num. | Compressible | Heat transfer |
| :--- | :---      | ---:  | ---: | ---:    | ---: | ---: | ---: | ---: |
| 15  | FPC         | 100   | 3D  | WALE     | Yes | [Homann](https://ntrs.nasa.gov/api/citations/19930093896/downloads/19930093896.pdf) | No | No |
| 16  | FPC         | 390   | 2D* | WALE     | Yes | [Thom](https://dx.doi.org/10.1098/rspa.1933.0146) | No | No |
| 17  | FPC         | 3,900 | 2D  | SSTIDDES | No  | [Ong](https://doi.org/10.1007/BF00189383)  | No | No |
| 18  | FPC         | 3,900 | 2D* | WALE     | Yes | [Ong](https://doi.org/10.1007/BF00189383) | No | No |
| 19  | FPC         | 9,000 | 2D* | WALE     | Yes | - | No | No |
| 20  | FPC         | 9,000 | 3D  | WALE     | Yes | - | No | No |
| 21  | JIMP, H/B=2 | 30,000| 2D | SSTCD     | No | [Zhe](https://asmedigitalcollection.asme.org/fluidsengineering/article/123/1/112/459689/Near-Wall-Measurements-for-a-Turbulent-Impinging) | No | Yes |
| 22  | JIMP, H/B=6 | 11,000| 2D | SSTCD     | No | [Huang](https://www.sciencedirect.com/science/article/pii/S0142727X2400198X) | No | Yes |
| 23  | JIMP, H/B=7    | 20,000| 2D | SSTLMCD   | No | [Zhe](https://asmedigitalcollection.asme.org/fluidsengineering/article/123/1/112/459689/Near-Wall-Measurements-for-a-Turbulent-Impinging) | No | Yes |
| 24  | JIMP, H/Dj=4   | 23,000| 2D | SSTLMCD+VC | Yes | [Lee](https://asmedigitalcollection.asme.org/heattransfer/article/126/4/554/464218/The-Effects-of-Nozzle-Diameter-on-Impinging-Jet) | No | Yes |
| 25  | JIMP, H/Dj=6   | 70,000| 2D | SSTLMCD+VC | Yes | [Yan](https://www.researchgate.net/publication/279909947_Effect_of_Reynolds_number_on_the_heat_transfer_distribution_from_a_flat_plate_to_an_impinging_jet) | No | Yes |
| 26  | JIMP, H/Dj=1   | 100,000| 2D | SSTLMCD+VC | Yes | [Fénot](https://www.sciencedirect.com/science/article/pii/S1290072917319440) | Yes | Yes |
| 27  | JIMP, H/Dj=5   | 100,000| 2D | SSTLMCD+VC | Yes | [Fénot](https://www.sciencedirect.com/science/article/pii/S1290072917319440)  | Yes | Yes |
| 28  | JIMP-C, H/Dj=2 | 23,000 | 2D | SSTLMCD+VC | Yes | [Huang](https://www.sciencedirect.com/science/article/pii/S0017931021010851) | No | Yes |
| 29  | FPNA, theta=0   | 300,000| 2D | SSTLMCD | No | [Gregory](https://www.semanticscholar.org/paper/Low-Speed-Aerodynamic-Characteristics-of-NACA-0012-Gregory-'reilly/c6149c61311559ef83b93ba9835e09170d27c530) | Yes | No |
| 30  | FPNA, theta=10   | 300,000| 2D | SSTLMCD | No | [Gregory](https://www.semanticscholar.org/paper/Low-Speed-Aerodynamic-Characteristics-of-NACA-0012-Gregory-'reilly/c6149c61311559ef83b93ba9835e09170d27c530) | Yes | No |
| 31  | JF   | - | 2D | laminar | No | [NACA](https://www.grc.nasa.gov/WWW/wind/valid/cdv/cdv.html) | Yes | No |
| 32  | Channel flow   | - | 3D | WALE | Yes | - | No | No |


**Geometries** <br>
<img width="952" height="176" alt="ML-geometry" src="https://github.com/user-attachments/assets/5285a58e-0a8e-4548-8264-28f696d36f53" />


<h2><strong>(2) Data portions in different cases</strong></h2>
<p align="center">
  <img src="https://github.com/user-attachments/assets/1a213b47-e441-4ed0-88d0-0bbb6586f571" width="45%"  />
  <img src="https://github.com/user-attachments/assets/ce06f2f8-89f5-423e-bb27-d70c4626f775" width="45%" alt="ML" />
</p>


<h2><strong>(3) Validation</strong></h2>


| I | Cases         | Re    | Dim. | Reference model     |
|:-:|:--------------|:-----:|:----:|:--------------------|
| 1 | JIMP, H/Dj=6  | 23000 | 2D   | SSTLMCD+VC          |
| 2 | FPC           | 120   | 2D   | WALE                |
| 3 | FPC           | 2000  | 2D   | SSTIDDES or WALE    |
| 4 | FPC           | 15000 | 2D   | WALE                |

**JIMP** <br>
<img src="https://github.com/user-attachments/assets/5a23772b-d16a-4010-b2e8-2e8db9fbd0a4" width="100%" />

**Flow past a cylinder** <br>
(1) Re=3,900 <br>
<img src="https://github.com/user-attachments/assets/81fdb495-096b-401d-b40f-c1f6db6c6ae2" width="100%" />

(2) Re=120, 2,000, 3,900, 15,000 <br>
<img  src="https://github.com/user-attachments/assets/d87121a4-5ed3-4295-876f-c6b15dda7199" width="100%" />

(3) Energy balance (Why we say this method is physics-informed)<br>
<img src="https://github.com/user-attachments/assets/96cb34a4-0e4c-4d5e-a18f-da0ad0c7302c" width="100%" />

(4) Solving properties <br>
<img src="https://github.com/user-attachments/assets/7c096866-c259-496e-ab0f-e85c176f972c" width="100%" />

**Note**: All the above information can be found in our manuscript submited in JFM Rapids


