
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long SIZE, MEASUREMENT;

float dp(long N, const float *pA, const float *pB) {
    float R = 0.0F;
    int j;
    for (j=0;j<N;j++)
        R += pA[j]*pB[j];
    return R;
}

int main(int argc, char** argv) {
    if (argc == 3) {
        SIZE = (long) strtol(argv[1], NULL, 10);
        MEASUREMENT = (long) strtol(argv[2], NULL, 10);
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

    long total_nanosecond = 0;
    long total_second = 0;
    float ans = 0;
    for (i = 0; i < MEASUREMENT; i++) {
        // lets rock
        struct timespec tik, tok;
        clock_gettime(CLOCK_MONOTONIC, &tik);
        float r = dp(SIZE, pA, pB); // return value is unused
        clock_gettime(CLOCK_MONOTONIC, &tok);
        total_second += tok.tv_sec - tik.tv_sec;
        total_nanosecond += tok.tv_nsec - tik.tv_nsec;
        ans += r;
    }
    printf("%ld, %ld\n", total_second, total_nanosecond);
    double total_time = (double) total_second + (double) total_nanosecond / 1e9;
    double avg_time = total_time / (double) MEASUREMENT;

    // output data
    double bandwidth = 2.0*(double) SIZE * sizeof(float) / avg_time / 1e9;
    double flop = 1.0*(double) SIZE / avg_time;
    printf("N: %ld  <T>: %f sec  B: %f GB/sec   F: %f FLOP/sec\n", SIZE, avg_time, bandwidth, flop);

}
