#include <iostream>

//const int H = 1024, W = 1024;
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

int main() {
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
    naive_convolution(input, filter, output);
    double checksum = calc_checksum(output, C, H, W);

    std::cout << checksum << std::endl;
    return 0;
}
