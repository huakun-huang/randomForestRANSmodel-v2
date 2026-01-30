
// Create a vector of inputs.
int kCount = 0;

// define an input arrary, 13 is the size
float arr[13];
int size = 13;

volScalarField k_ = mesh.lookupObject<const volScalarField>("k");
volScalarField omega = mesh.lookupObject<const volScalarField>("omega");
volSymmTensorField R = turbulence->R();
volScalarField nut = mesh.lookupObject<const volScalarField>("nut");

Info<<"create features for Deep learning net work"<<endl;
#include "createFeaturesNN.H"

Info<<"get features for Deep learning net work"<<endl;
#include "getInputFeaturesV2.H"

Info<<"output features for Deep learning net work"<<endl;
//volScalarField q12 = OMEGA/STRAIN;
mut = nut/(Uc*Lc);

#include "outputFeatures.H"

Info<<"random Forest Predict"<<endl;

system("python /home/hhk/OpenFOAM/PINNv/include/regressionSolverRANS.py");

Info<<"get the output results"<<endl;
{
    //read the data file
    FILE *dats = fopen("outputResult.txt", "r");
    double buffer=0;
    double critical = 2.;
    bool noTrue = false;
    if(dats!=NULL)
    {
        forAll(Pkf, celli)
        {
            fscanf(dats, "%lf", &buffer);
            Pkf[celli] = buffer;
            if(buffer>critical)
            {
                noTrue = true;  
            }
            fscanf(dats, "%lf", &buffer);
            Dkf[celli] = buffer;
            if(buffer>critical)
            {
                noTrue = true;  
            }
            fscanf(dats, "%lf", &buffer);
            Pwf[celli] = buffer;
            if(buffer>critical)
            {
                noTrue = true;  
            }
            fscanf(dats, "%lf", &buffer);
            Dwf[celli] = buffer;
            if(buffer>critical){
                noTrue = true;  
            }
            if(noTrue)
            {
                 Pkf[celli] = critical;
                 Dkf[celli] = critical;
                 Pwf[celli] = critical;
                 Dwf[celli] = critical;
            }
            noTrue = false;
        }
        fclose(dats);
    }
}

/*forAll(U, celli)
{
    std::vector<torch::jit::IValue> inputs;
    for(int ii=0; ii<size-2; ii++)
    {
        arr[ii] = 0;
        arr[ii] = inputFeatures[ii][celli];
    }
    arr[size-2] = mut[celli];
    arr[size-1] = q12[celli];
    
    torch::Tensor units = torch::from_blob(arr, {1,size}, torch::kFloat);
    //inputs.push_back(units);
    Info<<"here1"<<endl;
    at::Tensor output = model->forward(units);
    Info<<"here2"<<endl;
    float *parameters = output.data<float>();
    
    Pkf[celli] = parameters[0];
    Dkf[celli] = parameters[1];
    Pwf[celli] = parameters[2];
    Dwf[celli] = parameters[3];
}*/



