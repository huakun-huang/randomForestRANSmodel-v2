FOAM='/home/hhk/OpenFOAM/OpenFOAM-v2312/src/TurbulenceModels/turbulenceModels/lnInclude/'
cd $FOAM &&
ln -s PINNkOmegaSST.C &&
ln -s PINNkOmegaSST.H && 
ln -s PINNkOmegaSSTBase.C && 
ln -s PINNkOmegaSSTBase.H 

cd /home/hhk/OpenFOAM/OpenFOAM-v2312/src/TurbulenceModels &&
./Allwmake