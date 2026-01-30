        if(runTime.writeTime())
        {
            if(func==1)
            {
                #include "torchANN.H"
            }
            else if(func==0)
            {
                #include "torchRandomForest.H"
                for(int j=0; j<int(iteration.value()); j++)
                {
                    Info<< "Iteration = " << j << nl << endl;               
                    #include "UEqn.H"
                    #include "TEqn.H"
                    #include "pEqn.H"
                    laminarTransport.correct();
                    turbulence->correct(); 
                }
                //restore value
                Pkf = pkf0;
                Dkf = dkf0;
                Pwf = pwf0;
                Dwf = Dwf0;                  
            }          
        }