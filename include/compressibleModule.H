//--------------------------- read temperature ----------------------------------
volScalarField TRatio
(
      IOobject
      (
            "TRatio",
            latestTime,
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true
      ),
      mesh,
      dimensionedScalar("TRatio", dimless, 1.0)
);

{
    //reference temperature
    dimensionedScalar Tref
    (
         "Tref",
         dimensionSet(0, 0, 0, 1, 0, 0, 0),
         transportProperties.lookup("Tref")
    );
    
    Info<<"read temperature field?"<<endl;
    dimensionedScalar readT
    (
         "readT",
         dimensionSet(0, 0, 0, 0, 0, 0, 0),
         transportProperties.lookup("readT")
    );
    if( int(readT.value())==1)
    {
        Info<<"read temperature"<<endl;
        volScalarField T0
        (
            IOobject
           (
                "T",
                latestTime,
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                true
            ),
           mesh
        );
        forAll(TRatio, celli)
            TRatio[celli] = T0[celli]/Tref.value();
    }
   else
   {
       Info<<"incompressible flow"<<endl;
       TRatio = 1.0;
   }
}
Info<<"read temperature done"<<endl;