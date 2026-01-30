Info<<"Output region is:"<<endl;
Info<<"("<<Xmin<<" "<<Ymin<<" "<<Zmin<<")"<<endl;
Info<<"("<<Xmax<<" "<<Ymax<<" "<<Zmax<<")"<<endl;
FILE *data = fopen("inputFeatures.txt", "w");
//volScalarField q12 = OMEGA/STRAIN;
pointField points_ = mesh.C();
if(data!=NULL)
{
    forAll(U, celli)
    {
        double x = points_[celli][0];
        double y = points_[celli][1];
        double z = points_[celli][2];
        if( x>Xmin.value() && x<Xmax.value())
        {
            if( y>Ymin.value() && y<Ymax.value())
            {
                if( z>Zmin.value() && z<Zmax.value())
                {
                      for(int i = 0; i<inputFeaturesSize; i++)
                      {
                           fprintf(data, "%g\t", inputFeatures[i][celli]);
                      }
                     fprintf(data, "%g\t", mut[celli]);
                     fprintf(data, "%g\t", OMEGA[celli]/std::max(STRAIN[celli], 1e-10));

                    //compressibleModule
                    fprintf(data, "%g\t", TRatio[celli]);

                    fprintf(data, "\n");
                }
            }
        }
    }
    fclose(data);
}

Info<<"Output features: "<<inputFeaturesSize+3<<endl;