# Q1

On **K80** GPU, w/o data loading, it takes

| Batch size           | 32     | 128    | 512    |
| -------------------- | ------ | ------ | ------ |
| **Train** Time (sec) | 96.315 | 70.056 | 63.323 |

When batch size is larger, it takes shorter time to train, because large batch size means higher working volumn for single GPU and less CPU-GPU transfer times, so we can make fully use of GPU.



# Q2

|       | Batch size | 32 / gpu | Batch size | 128 / gpu | Batch size | 512 / gpu |
| ----- | ---------- | -------- | ---------- | --------- | ---------- | --------- |
|       | Time       | Speedup  | Time       | Speedup   | Time       | Speedup   |
| 1 GPU | 98.611     | 1        | 71.038     | 1         | 64.194     | 1         |
| 2 GPU | 90.064     | 1.09     | 50.828     | 1.40      | 38.528     | 1.67      |
| 4 GPU | 90.334     | 1.09     | 35.331     | 2.01      | 24.339     | 2.64      |



We are measuring weak scaling because workload (batch size) for each processor (GPU) stays constant. 

Strong scaling speed up is obviously worse than weak scaling with fixed batch size. In strong scaling, 1 gpu with 128 batch size is scaled into 4 gpu with 32 batch size. From the datapoint above, **71.038 < 90.334**, so weak scaling is better. The reason lies in that for strong scaling we must spend more time on communication with small batch size.



# Q3

## Q3.1



|       | Batch size | 32       | Batch size | 128    | Batch size | 512   |
| ----- | ---------- | -------- | ---------- | ------ | ---------- | ----- |
|       | Compute    | Comm     | Compute    | Comm   | Compute    | Comm  |
| 2-GPU | 48.1575    | 41.906   | 35.028     | 15.800 | 31.6615    | 6.86  |
| 4-GPU | 24.07875   | 66.25525 | 17.514     | 17.817 | 15.83075   | 8.508 |

Based on assumtion that computation intensity for single GPU is the same as in multiple GPUs mode.

For $$n$$ GPU,

Computation time for n GPU = Computation time for 1 GPU / n

Communication for n GPU = Total time for n GPU - Computation time for n GPU

​											    =  Total time for n GPU  - (Computation time for 1 GPU / n)



## Q3.2

From paper **"Bandwidth optimal all-reduce algorithms for clusters of workstations"** we have formula:

$$Data\ bytes\ to\ move = 2 \times (N-1) \times \frac{X}{N} \times itsize$$ 

Where it size is float32, i.e. 4 bytes, X is parameters, i.e. 11173962, N is the gpu number

in total, 

Bandwith = $$ \frac{N-1}{N} \times  $$