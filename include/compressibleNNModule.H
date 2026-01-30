//--------------------------- read temperature ----------------------------------
volScalarField TRatio
(
      IOobject
      (
            "TRatio",
            runTime.timeName(),
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
    dimensionedScalar readT
    (
         "readT",
         dimensionSet(0, 0, 0, 0, 0, 0, 0),
         transportProperties.lookup("readT")
    );
    if( int(readT.value())==1)
    {
        volScalarField T0
        (
            IOobject
           (
                "T",
                runTime.timeName(),
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
       TRatio = 1.0;
   }
}