#include <iostream>

int main() {
    long N = 30000000;
    // create arrays
    double* pA = (double*) malloc(N*sizeof(double));
    double* pB = (double*) malloc(N*sizeof(double));

    // initialize
    long i;
    for (i = 0; i < N; i++) {
        pA[i] = 1.0f;
        pB[i] = 1.0f;
    }

    double R = 0.0F;
    int j;
    for (j=0;j<N;j++)
        R += pA[j]*pB[j];

    printf("%f, %f", (double) N, R);

}
