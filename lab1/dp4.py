import numpy as np
import sys
from timeit import default_timer as timer
import math


def dp(N, A, B):
    R = 0.0
    for j in range(0, N):
        R += A[j] * B[j]
    return R


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("usage: ./program <size> <measurements>")
        exit(0)
    N = int(sys.argv[1])
    repetition = int(sys.argv[2])

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    time_taken = 0
    for i in range(0, repetition):
        start = timer()
        dp(N, A, B)
        end = timer()
        if i >= repetition/2:
            time_taken += end - start

    time_taken /= math.ceil(repetition/2)
    bandwidth = 2 * 4 * N / time_taken / 1e9
    flop = 2 * N / time_taken

    print("N: %d  <T>: %f sec  B: %f GB/sec   F: %f FLOP/sec" % (N, time_taken, bandwidth, flop))
