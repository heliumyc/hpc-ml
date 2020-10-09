#include <iostream>
#include <mkl_cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float dpunroll(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j=0;j<N;j+=4)
        R += pA[j]*pB[j] + pA[j+1]*pB[j+1] + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3];
    return R;
}

float bdp(long N, float *pA, float *pB) {
    float R = cblas_sdot(N, pA, 1, pB, 1);
    return R;
}

int main() {
    long N = 300000000;
    // create arrays
    float* pA = (float*) malloc(N*sizeof(float));
    float* pB = (float*) malloc(N*sizeof(float));

    // initialize
    long i;
    for (i = 0; i < N; i++) {
        pA[i] = 1.0f;
        pB[i] = 1.0f;
    }

    float R = bdp(N, pA, pB);

    printf("%f, %f", (float) N, R);

}
