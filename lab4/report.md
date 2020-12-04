# Q1

On **K80** GPU, w/o data loading, it takes

| Batch size           | 32      | 128     | 512     |
| -------------------- | ------- | ------- | ------- |
| **Train** Time (sec) | 162.399 | 161.786 | 170.099 |



In fact, I think 4 fold (32,128,512) parameter range does not reveal too much, so I tested with smaller granularity.

| Batch size | Time(s) |
| ---------- | ------- |
| 16         | 494.993 |
| 32         | 162.399 |
| 64         | 84.342  |
| 128        | 161.786 |
| 256        | 177.793 |
| 512        | 170.099 |



We can find that when batch size is relatively small (say 16), larger batch size will boost training drastically until it reaches its best at 64. When it continues increasing, the time it takes bounces back. 

The reason for this, I believe, is that when batch size is small, GPU units are not fully used and most are still idle so it takes longer to train all batches, thus increasing batch size will make fully use of GPU. But after reaching the optimal size, GPU might not able to deal with one batch at a time so scheduling could slow down the total time. Also, cache missing and bank conclict could contribute to slower running time.



# Q2

|       | Batch size | 32 / gpu | Batch size | 128 / gpu | Batch size | 512 / gpu |
| ----- | ---------- | -------- | ---------- | --------- | ---------- | --------- |
|       | Time       | Speedup  | Time       | Speedup   | Time       | Speedup   |
| 1 GPU | 164.678    | 1        | 162.897    | 1         | 170.885    | 1         |
| 2 GPU | 90.064     | 1.828    | 50.828     | 3.204     | 38.528     | 4.435     |
| 4 GPU | 92.328     | 1.784    | 36.478     | 4.466     | 24.060     | 7.102     |



# Q3

## Q3.1



|       | Batch size | 32   | Batch size | 128  | Batch size | 512  |
| ----- | ---------- | ---- | ---------- | ---- | ---------- | ---- |
|       | Compute    | Comm | Compute    | Comm | Compute    | Comm |
| 2-GPU |            |      |            |      |            |      |
| 4-GPU |            |      |            |      |            |      |







data point

1 gpu 2048 = 121.080