#include "mex.h"
#include <math.h>
#include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
# include "ranlib.h"
# include "rnglib.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    // Z = mnrnd(updf');
   
    double * updf;
    int N, D;
    updf = mxGetPr(prhs[0]);
    D = (int) mxGetM(prhs[0]);
    N = (int) mxGetN(prhs[0]);
    
    int n;
    
    //plhs[0] = mxCreateDoubleMatrix(D,N,mxREAL);// output lambda
    plhs[0] = mxCreateNumericMatrix(D, N, mxUINT32_CLASS, mxREAL);
    int * rnum = (int *)mxGetPr(plhs[0]);
    for(n = 0; n < N; n ++)
    {
        genmul2(1, &(updf[n * D]), D, &(rnum[n * D]));
        //mnrnd2(&(updf[n * D]), D, &(rnum[n * D]));
    }
}