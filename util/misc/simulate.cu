#include <iostream>
#include <unistd.h>
using namespace std;

#define CUDA_CALL(func)                         \
 {                                              \
    cudaError_t err = func;                     \
    if(err != cudaSuccess) {                    \
        cout << __FILE__ << ":" << __LINE__     \
             << " " << #func << " "             \
             << cudaGetErrorString(err)         \
             << " errnum " << err;              \
        exit(EXIT_FAILURE);                     \
    }                                           \
 }

int main(int argc, char **argv) {
    int dev = std::stoi(argv[1]);
    size_t bytes = std::stoull(argv[2]) * 1024 * 1024;
    void* ptr_dev;
    CUDA_CALL(cudaSetDevice(dev));
    CUDA_CALL(cudaMalloc(&ptr_dev, bytes));

    while(1) {
        sleep(10);
    }
}