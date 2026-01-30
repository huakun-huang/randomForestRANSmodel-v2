Info<<"load the Pt model for CNN"<<endl;
char model[] = {"/mnt/d/machineLearningVersion/CNN/ReNet/hhk_PINNmodel.pt"};
    
torch::jit::script::Module module;
try
{
        module = torch::jit::load(model);
        module.eval();
        Info<<"load the Pt model done"<<endl;
}
catch (const c10::Error & e)
{
        std::cerr << "error loading the model\n";
        return -1;
}