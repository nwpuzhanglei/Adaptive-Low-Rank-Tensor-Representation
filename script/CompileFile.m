function [] = CompileFile()

mex mnrndC.cpp ranlib.cpp rnglib.cpp
mex SampleLambdaC.cpp
mex SampleUC.cpp

end