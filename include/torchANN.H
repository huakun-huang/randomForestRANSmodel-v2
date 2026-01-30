// Create a vector of inputs.
int kCount = 0;

// define an input arrary, 59 is the size
int size = 59;

volScalarField k_ = turbulence->k();
volScalarField omega = turbulence->omega();
volSymmTensorField R = turbulence->R();
volScalarField nut = turbulence->nut();

Info<<"Create features for Deep learning net work"<<endl;
#include "createFeaturesNN.H"

Info<<"Get features for Deep learning net work"<<endl;
#include "getInputFeatures.H"

Info<<"Output features for Deep learning net work"<<endl;
volScalarField q12 = OMEGA/STRAIN;
Info<<"Output q12"<<endl;
mut = nut/(Uc*Lc);

std::vector<torch::jit::IValue> inputs(1);
torch::NoGradGuard no_grad;

int batch_size = U.size();  
torch::Tensor batch_tensor = torch::empty({batch_size, size}, torch::kFloat32);

Info<< "Trans the input features to tensor" <<endl;
auto batch_data = batch_tensor.accessor<float, 2>();
forAll(U, celli)
{
    for(int ii=0; ii<size-3; ii++)
    {
         batch_data[celli][ii] = inputFeatures[ii][celli];
    }
    batch_data[celli][size-3] = mut[celli];
    batch_data[celli][size-2] = std::max(q12[celli], 1e-10);
    batch_data[celli][size-1] = TRatio[celli];
}
    
Info<< "Infer the results" <<endl;
inputs[0]=batch_tensor;
at::Tensor batch_output = module.forward(inputs).toTensor();
auto output_data = batch_output.accessor<float, 2>();

Info<< "Reconstruct the parameters" <<endl;
forAll(U, celli)
{    
    Pkf[celli] = std::max(output_data[celli][0]*CRITICLE, 0.0f);
    Dkf[celli] = std::max(output_data[celli][1]*CRITICLE, 0.0f);
    Pwf[celli] = std::max(output_data[celli][2]*CRITICLE, 0.0f);
    Dwf[celli] = std::max(output_data[celli][3]*CRITICLE, 0.0f);
}

Info<<"Predict done for Deep learning net work"<<endl;


