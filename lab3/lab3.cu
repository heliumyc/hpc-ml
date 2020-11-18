#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cuda.h>
#include <cudnn.h>

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

//const int H = 1024, W = 1024;
const int H = 4, W = 4;
const int C = 3, FW = 3, FH = 3, K = 3;
const int P = 1;
const int H0 = H + 2 * P;
const int W0 = W + 2 * P;
const int INPUT_SIZE = C * H * W;
const int FILTER_SIZE = C * K * FW * FH;
const int OUTPUT_SIZE = K * H * W;
const int INPUT_PADDED_SIZE = C * H0 * W0;

template<class T>
T &at(T *tensor, int c, int i, int j, int height, int width) {
    return tensor[c * height * width + i * width + j];
}

template<class T>
T &at(T *tensor, int k, int c, int i, int j, int layer, int height, int width) {
    return tensor[k * layer * height * width + c * height * width + i * width + j];
}

void init_input(double *input) {
    // real input
    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                at(input, c, x, y, H, W) = c * (x + y);
            }
        }
    }
}

void clear_output(double *output) {
    std::fill(output, output+OUTPUT_SIZE, 0);
}

void add_padding(double *raw, double *padded) {
    // padding, p=1, set to zero
    for (int c = 0; c < C; c++) {
        // top and bottom
        for (int j = 0; j < W0; j++) {
            at(padded, c, 0, j, H0, W0) = 0;
            at(padded, c, H0 - 1, j, H0, W0) = 0;
        }
        // left and right
        for (int i = 0; i < H0; i++) {
            at(padded, c, i, 0, H0, W0) = 0;
            at(padded, c, i, W0 - 1, H0, W0) = 0;
        }
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                at(padded, c, x+1, y+1, H0, W0) = at(raw, c, x, y, H, W);
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

void print_mat(double *mat, int channel, int height, int width) {
    for (int c = 0; c < channel; c++) {
        for (int i = 0 ; i < height; i++) {
            for (int j = 0 ; j < width; j++) {
                std::cout << at(mat, c, i, j, height, width) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void naive_convolution(double *input, double *filter, double *output) {
    for (int k = 0; k < K; k++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                // conv sum
                double sum = 0;
                for (int c = 0; c < C; c++) {
                    for (int j = 0; j < FH; j++) {
                        for (int i = 0; i < FW; i++) {
                            sum += at(filter, k, c, FW - 1 - i, FH - 1 - j, C, FW, FH) *
                                   at(input, c, x + i, y + j, H0, W0);
                        }
                    }
                }
                at(output, k, x, y, H, W) = sum;
            }
        }
    }
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

void run_cudnn(double *input, double *filter, double *output) {

    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    // define input descriptor
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    // desc, format, data type, channels, height, width
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));
    // copy data from host to device
    double *input_d;
    CUDA_CALL(cudaMalloc(&input_d, INPUT_SIZE * sizeof(double)));
    CUDA_CALL(cudaMemcpy(input_d, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    // define filter descriptor
    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW));
    double *filter_d;
    CUDA_CALL(cudaMalloc(&filter_d, FILTER_SIZE * sizeof(double)));
    CUDA_CALL(cudaMemcpy(filter_d, filter, FILTER_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    // define output descriptor
    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));
    double *output_d;
    CUDA_CALL(cudaMalloc(&output_d, OUTPUT_SIZE* sizeof(double)));
    CUDA_CALL(cudaMemcpy(output_d, output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    // define conv pre action
    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    // desc, pad h, pad w, vertical stride, horizontal stride, dilation height, dilation width, mode, data type
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_descriptor, P, P, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

    // define algo
    cudnnConvolutionFwdAlgo_t algorithm;
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor, filter_descriptor, conv_descriptor, output_descriptor,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algorithm));

    // workspace
    size_t ws_size;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, filter_descriptor, conv_descriptor, output_descriptor, algorithm, &ws_size));
    double *ws_data;
    CUDA_CALL(cudaMalloc(&ws_data, ws_size));


    // perform conv !!!!!!!!1
    double alpha = 1. , beta = 0.;
    CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, input_d, filter_descriptor, filter_d,
                                       conv_descriptor, algorithm, ws_data, ws_size, &beta, output_descriptor, output_d));

    // copy back
    CUDA_CALL(cudaMemcpy(output, output_d, OUTPUT_SIZE, cudaMemcpyDeviceToHost));

    // finalizing
    CUDA_CALL(cudaFree(ws_data));
    CUDA_CALL(cudaFree(output_d));
    CUDA_CALL(cudaFree(filter_d));
    CUDA_CALL(cudaFree(input_d));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    CUDNN_CALL(cudnnDestroy(cudnn));

}

int main() {

    // malloc
    auto *input = (double *) malloc(sizeof(double) * INPUT_SIZE);
    auto *input_padded = (double *) malloc(sizeof(double) * INPUT_PADDED_SIZE);
    auto *filter = (double *) malloc(sizeof(double) * FILTER_SIZE);
    auto *output = (double *) malloc(sizeof(double) * OUTPUT_SIZE);
    // init zero
    std::fill(input, input + INPUT_SIZE, 0);
    std::fill(input_padded, input_padded + INPUT_PADDED_SIZE, 0);
    std::fill(filter, filter + FILTER_SIZE, 0);
    std::fill(output, output + OUTPUT_SIZE, 0);

    // init
    init_input(input);
    add_padding(input, input_padded);
    init_filter(filter);

    double checksum = 0;

    // naive conv cpu mode
    naive_convolution(input_padded, filter, output);
    checksum = calc_checksum(output, K, H, W);
    std::cout << checksum << std::endl;

    // cuda

    // cuda tiled

    // cuDNN
    clear_output(output);
    print_mat(output, K, H, W);
    run_cudnn(input, filter, output);
    checksum = calc_checksum(output, K, H, W);
    std::cout << checksum << std::endl;
    print_mat(output, K, H, W);


//    print_mat(input, C, H, W);
//    print_mat(input_padded, C, H0, W0);
    return 0;
}
