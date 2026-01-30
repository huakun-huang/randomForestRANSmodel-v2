#include <cstdlib> // for getenv()
#include <vector>
#include <string>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>

double readCritical()
{
    std::string file = "/mnt/d/Codes/PINNv/include/Critical.H";
    FILE *data = fopen(file.c_str(), "r");
    if(data==NULL)
    {
        printf("Cannot find the file: %s\n", file.c_str());
    }
    char buffer[50];
    double critical=0;
    fscanf(data, "%s", buffer);
    printf("%s  ", buffer);
    fscanf(data, "%s", buffer);
    printf("%s  ", buffer);
    fscanf(data, "%s", buffer);
    critical = std::stof(buffer);
    printf("%lf  ", critical);
    fclose(data);
    return critical;
}

double normalization(double up, double down)
{
    return up/max(fabs(up)+fabs(down), VSMALL);
}

//-define the cross product for second order tensor
volTensorField MycrossProduct(volTensorField a, volVectorField b)
{
    volTensorField c = a*0;
    forAll(a, celli)
    {
         c[celli].xx() = b[celli].y()*a[celli].zx()+-b[celli].z()*a[celli].yx();
         c[celli].xy() = b[celli].y()*a[celli].zy()-b[celli].z()*a[celli].yy(); 
         c[celli].xz() = b[celli].y()*a[celli].zz()-b[celli].z()*a[celli].yz();
         c[celli].yx() = -b[celli].x()*a[celli].zx()+b[celli].z()*a[celli].xx();
         c[celli].yy() = -b[celli].x()*a[celli].zy()+b[celli].z()*a[celli].xy();
         c[celli].yz() = -b[celli].x()*a[celli].zz()+b[celli].z()*a[celli].xz();
         c[celli].zx() = b[celli].x()*a[celli].yx()-b[celli].y()*a[celli].xx();
         c[celli].zy() = b[celli].x()*a[celli].yy()-b[celli].y()*a[celli].xy();
         c[celli].zz() = b[celli].x()*a[celli].yz()-b[celli].y()*a[celli].xz();
    }
    return c;
}

volTensorField normTensorField(volTensorField a, volTensorField b, volScalarField c)
{
    forAll(a, celli)
    {
        a[celli].xx() =  normalization(b[celli].xx(), c[celli]);
        a[celli].xy() =  normalization(b[celli].xy(), c[celli]);
        a[celli].xz() =  normalization(b[celli].xz(), c[celli]);

        a[celli].yx() =  normalization(b[celli].yx(), c[celli]);
        a[celli].yy() =  normalization(b[celli].yy(), c[celli]);
        a[celli].yz() =  normalization(b[celli].yz(), c[celli]);
        
        a[celli].zx() =  normalization(b[celli].zx(), c[celli]);
        a[celli].zy() =  normalization(b[celli].zy(), c[celli]);
        a[celli].zz() =  normalization(b[celli].zz(), c[celli]);
    }
    return a;
}

bool fileExists(const std::string& path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

std::string findFileInPaths(const std::vector<std::string>& paths, const std::string& filename)
{
    for(const auto& dir:paths)
    {
        std::string fullPath = dir + "/"+filename;
        if(fileExists(fullPath))
            return fullPath;
    }
    return " ";
}

std::string getPythonPath(const std::string& files)
{
    std::vector<std::string> paths;
    const char *pythonPath = getenv("PINNPYTHONPATH");
    
    if (pythonPath != nullptr)
    {
        std::string pathStr(pythonPath);
        std::istringstream iss(pathStr);
        std::string path;
        
        while(getline(iss, path, ':'))
        {
            if (!path.empty())
            {
                paths.push_back(path);
            }
        }
    }
    
    std:: string filePath = findFileInPaths(paths, files);
    return filePath;
}

void runPythonFile(const std::string file)
{
    std::string fullPath = getPythonPath(file);
    try
    {
        std::string cmd = "python  "+fullPath;
        system(cmd.c_str());
    }
    catch (...) 
    {
        std::string cmd = "python3  "+fullPath;
        system(cmd.c_str());
    }
}