#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cudnn.h>

//////////////////////////
const int H = 4, W = 4;
const int C = 3, FW = 3, FH = 3, K = 64;
const int P = 1;
const int H0 = H + 2 * P;
const int W0 = W + 2 * P;

template<class T>
T &at(T *tensor, int c, int i, int j, int height, int width) {
    return tensor[c * height * width + i * width + j];
}

template<class T>
T &at(T *tensor, int k, int c, int i, int j, int layer, int height, int width) {
    return tensor[k * layer * height * width + c * height * width + i * width + j];
}

void init_input_tensor(double *tensor) {
    // padding, p=1, set to zero
    for (int c = 0; c < C; c++) {
        // top and bottom
        for (int j = 0; j < W0; j++) {
            at(tensor, c, 0, j, H0, W0) = 0;
            at(tensor, c, H0 - 1, j, H0, W0) = 0;
        }
        // left and right
        for (int i = 0; i < H0; i++) {
            at(tensor, c, i, 0, H0, W0) = 0;
            at(tensor, c, i, W0 - 1, H0, W0) = 0;
        }
    }
    // real tensor
    for (int c = 0; c < C; c++) {
        for (int x = 1; x < H0 - 1; x++) {
            for (int y = 1; y < W0 - 1; y++) {
                at(tensor, c, x, y, H0, W0) = c * (x + y);
            }
        }
    }
}

void init_filter(double *filter) {
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FH; i++) {
                for (int j = 0; j < FW; j++) {
                    at(filter, k, c, i, j, C, FH, FW) = (c + k) * (i + j);
                }
            }
        }
    }
}
//////////////////////////

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

__global__ void dev_const(double *px, double k) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    px[tid] = k;
}

__global__ void dev_iota(double *px) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    px[tid] = tid;
}

double calc_checksum(double *tensor, int layer, int height, int width) {
    double sum = 0;
    for (int c = 0; c < layer; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                sum += at(tensor, c, i, j, height, width);
            }
        }
    }
    return sum;
}




int main() {

    // init input and filter data
    int input_size = C * H0 * W0;
    int filter_size = C * K * FW * FH;
    int output_size = K * H * W;
    double *input = (double *) malloc(sizeof(double) * input_size);
    double *filter = (double *) malloc(sizeof(double) * filter_size);
    double *output = (double *) malloc(sizeof(double) * output_size);
    // init zero
    std::fill(input, input + input_size, 0);
    std::fill(filter, filter + filter_size, 0);
    std::fill(output, output + output_size, 0);
    init_input_tensor(input);
    init_filter(filter);



    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    // input
    const int in_n = 1;
    const int in_c = 3;
    const int in_h = 4;
    const int in_w = 4;

    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, in_n, in_c, in_h, in_w));

    double *in_data;
    CUDA_CALL(cudaMalloc(
            &in_data, in_n * in_c * in_h * in_w * sizeof(double)));

    // filter
    const int filter_k = 64;
    const int filter_c = 3;
    const int filter_h = 3;
    const int filter_w = 3;
    std::cout << "filter_k: " << filter_k << std::endl;
    std::cout << "filter_c: " << filter_c << std::endl;
    std::cout << "filter_h: " << filter_h << std::endl;
    std::cout << "filter_w: " << filter_w << std::endl;
    std::cout << std::endl;

    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
            filter_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW,
            filter_k, filter_c, filter_h, filter_w));

    double *filter_data;
    CUDA_CALL(cudaMalloc(&filter_data, filter_k * filter_c * filter_h * filter_w * sizeof(double)));

    // convolution
    const int pad_h = 1;
    const int pad_w = 1;
    const int str_h = 1;
    const int str_w = 1;
    const int dil_h = 1;
    const int dil_w = 1;
    std::cout << std::endl;

    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

    // output
    int out_n;
    int out_c;
    int out_h;
    int out_w;

    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_descriptor, filter_desc, &out_n, &out_c, &out_h, &out_w));

    std::cout << "out_n: " << out_n << std::endl;
    std::cout << "out_c: " << out_c << std::endl;
    std::cout << "out_h: " << out_h << std::endl;
    std::cout << "out_w: " << out_w << std::endl;
    std::cout << std::endl;

    cudnnTensorDescriptor_t out_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, out_n, out_c, out_h, out_w));

    double *out_data;
    CUDA_CALL(cudaMalloc(&out_data, out_n * out_c * out_h * out_w * sizeof(double)));

    // algorithm
    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor, filter_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

    // workspace
    size_t ws_size;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, filter_desc, conv_desc, out_desc, algo, &ws_size));

    double *ws_data;
    CUDA_CALL(cudaMalloc(&ws_data, ws_size));

    std::cout << "Workspace size: " << ws_size << std::endl;
    std::cout << std::endl;

    // perform
    double alpha = 1. , beta = 0.;
    dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
    dev_const<<<filter_w * filter_h, filter_k * filter_c>>>(filter_data, 1.f);
    CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, in_data, filter_desc, filter_data,
            conv_desc, algo, ws_data, ws_size, &beta, out_desc, out_data));

    // finalizing
    CUDA_CALL(cudaFree(ws_data));
    CUDA_CALL(cudaFree(out_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDA_CALL(cudaFree(filter_data));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDA_CALL(cudaFree(in_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroy(cudnn));
    return 0;
}