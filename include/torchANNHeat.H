
char model[] = {"/mnt/d/machineLearning/hhk_PINNmodel.pt"};
// load the model
torch::jit::script::Module module;
try
{
    module = torch::jit::load(model);
}
catch (const c10::Error & e)
{
    std::cerr << "error loading the model\n";
    return -1;
}

// Create a vector of inputs.
int kCount = 0;

// define an input arrary, 13 is the size
float arr[13];
int size = 13;

volScalarField k_ = turbulence->k();
volScalarField omega = turbulence->omega();
volSymmTensorField R = turbulence->R();
volScalarField nut = turbulence->nut();

Info<<"create features for Deep learning net work"<<endl;
#include "createFeaturesNN.H"

Info<<"get features for Deep learning net work"<<endl;
#include "getInputFeaturesV2.H"

Info<<"output features for Deep learning net work"<<endl;
volScalarField q12 = OMEGA/STRAIN;
mut = nut/(Uc*Lc);

std::vector<torch::jit::IValue> inputs(1);
forAll(U, celli)
{
    for(int ii=0; ii<size-2; ii++)
    {
        arr[ii] = 0;
        arr[ii] = inputFeatures[ii][celli];
    }
    arr[size-2] = mut[celli];
    arr[size-1] = q12[celli];
    
    torch::jit::IValue units = torch::from_blob(arr, {1,size});
    //auto reshape_unit = units.toTensor().view({-1, 1, 1, size});
    inputs[0]=units;
    at::Tensor output = module.forward(inputs).toTensor();
    float *parameters = output.data<float>();
    
    Pkf[celli] = parameters[0];
    Dkf[celli] = parameters[1];
    Pwf[celli] = parameters[2];
    Dwf[celli] = parameters[3];
}



