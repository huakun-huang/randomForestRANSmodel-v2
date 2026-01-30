    IOdictionary transportProperties
    (
       IOobject
       (
           "transportProperties",
           runTime.constant(),
           mesh,
           IOobject::MUST_READ_IF_MODIFIED,
           IOobject::NO_WRITE
       )
    );

    dimensionedScalar iteration
    (
       "iteration",
       dimless,
       transportProperties.lookupOrDefault("iteration", 4)
    );

    // method 0: 随机森林
    // method 1: CNN
    dimensionedScalar method
    (
       "method",
       dimless,
       transportProperties.lookupOrDefault("method", 0)
    );
    
    int func = int( method.value() );
    if( func==0)
        printf("Using the random forest method\n");
    else if(func==1)
        printf("Using the CNN method\n");

    volScalarField pkf0 = Pkf;
    volScalarField dkf0 = Dkf;
    volScalarField pwf0 = Pwf;
    volScalarField Dwf0 = Dwf;

     timeControl tControl(runTime);