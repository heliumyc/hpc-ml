/**
 *
 * USING FLOAT PRECISION WHEN N IS LARGE WOULD CAUSE SERIOUS ERROR
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

long SIZE, MEASUREMENT;

float dp(long N, float *pA, float *pB) {
    float R = 0.0F;
    int j;
    for (j=0;j<N;j++)
        R += pA[j]*pB[j];
    return R;
}

int main(int argc, char** argv) {
    if (argc == 3) {
        SIZE = strtol(argv[1], NULL, 10);
        MEASUREMENT = strtol(argv[2], NULL, 10);
    } else {
        printf("usage: ./program <size> <measurements>\n");
        exit(0);
    }

    // create arrays
    float* pA = (float*) malloc(SIZE*sizeof(float));
    float* pB = (float*) malloc(SIZE*sizeof(float));

    // initialize
    long i;
    for (i = 0; i < SIZE; i++) {
        pA[i] = 1.0f;
        pB[i] = 1.0f;
    }

    double time_taken = 0;
    for (i = 0; i < MEASUREMENT; i++) {
        // lets rock
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        float r = dp(SIZE, pA, pB); // return value is unused
        clock_gettime(CLOCK_MONOTONIC, &end);

        if (i >= MEASUREMENT/2) {
            time_taken += (double) (end.tv_sec - start.tv_sec) + (double) (end.tv_nsec - start.tv_nsec) * 1e-9;
        }

//        if (fabsf((float) SIZE - r)/(float)SIZE > 1e-6) {
//            printf("error of calculation: suppose: %f, found: %f\n", (float) SIZE, r);
//            exit(0);
//        }
        // simple make sure dp function call is not optimized by gcc
        if (r < 0) {
            printf("error of calculation: suppose: %f, found: %f\n", (float) SIZE, r);
            exit(0);
        }
    }
    long count = MEASUREMENT % 2 ? MEASUREMENT/2 : MEASUREMENT/2+1;
    double avg_time = time_taken / (double) count;

    // output data
    double bandwidth = 2.0*(double) SIZE * sizeof(float) / avg_time / 1e9;
    double flop = 2.0*(double) SIZE / avg_time;
    printf("N: %ld  <T>: %f sec  B: %f GB/sec   F: %f FLOP/sec\n", SIZE, avg_time, bandwidth, flop);

    return 0;
}
