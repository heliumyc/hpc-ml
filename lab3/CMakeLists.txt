cmake_minimum_required(VERSION 3.17)
project(lab3 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(lab3 lab3.cu)

set_target_properties(
        lab3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)