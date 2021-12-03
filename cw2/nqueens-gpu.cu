#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <stack>
#include <thread>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sstream>

//#include "gpuErrchk.h"

#define N_MAX 12

// credit - https://stackoverflow.com/a/32531982/11136104
#include "cuda_runtime_api.h"
int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 7: // Volta and Turing
        if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 8: // Ampere
        if (devProp.minor == 0) cores = mp * 64;
        else if (devProp.minor == 6) cores = mp * 128;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

__device__ bool boardIsValid(const int* gameBoard, const int N)
{
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (gameBoard[i] - gameBoard[j] == i - j || gameBoard[i] - gameBoard[j] == j - i || gameBoard[i] == gameBoard[j])
                return false;
    return true;
}

__global__ void getPermutations(const int N, const int O, int* d_solutions, int* d_num_solutions) {
    int column = threadIdx.x + blockIdx.x * blockDim.x;
    if (column >= O)
        return;

    int gameBoard[N_MAX];
    for (int i = 0; i < N_MAX; i++)
        gameBoard[i] = -1;

    for (int i = 0; i < N; i++) {
        //printf("%d %d %d %d\n", N, threadIdx.x + blockIdx.x * blockDim.x, column, column % N);
        gameBoard[i] = column % N;
        column /= N;
    }

    __syncthreads();

    if (boardIsValid(gameBoard, N)) {
        int index = atomicAdd(d_num_solutions, 1);
        for (int i = 0; i < N; i++)
            d_solutions[N * index + i] = gameBoard[i] + 1; //"+1" so that we can tell later which indexes of "d_solutions" are empty using 0
        //printf("%d\n", d_permutations[index][0]);
    }

    __syncthreads();
}

void calculateSolutions(const int N, std::vector<std::vector<int>>* solutions, int* h_num_solutions)
{
    *h_num_solutions = 0;
    int* d_solutions = nullptr;
    int* d_num_solutions = nullptr;

    int O = pow(N, N);

    size_t solutions_mem = pow(N, 5) * sizeof(int*); // N^5 is an estimation of the amount of solutions for size N (^5 because N_MAX^4 (12^4) is enough to hold all the solutions for a 12x12 board and to store N columns for that board that would make it N^5)
    cudaMalloc((void**)&d_solutions, solutions_mem);
    cudaMalloc((void**)&d_num_solutions, sizeof(int));

    cudaMemcpy(d_num_solutions, h_num_solutions, sizeof(int), cudaMemcpyHostToDevice);
    
    getPermutations<<<(O + 512 - 1) / 512, 512>>>(N, O, d_solutions, d_num_solutions);

    cudaMemcpy(h_num_solutions, d_num_solutions, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_num_solutions);
    
    int* h_solutions = (int*)malloc(solutions_mem);
    cudaMemcpy(h_solutions, d_solutions, solutions_mem, cudaMemcpyDeviceToHost);
    cudaFree(d_solutions);

    for (int i = 0; i < *h_num_solutions; i++) {
        if (h_solutions[N * i] != NULL) {
            std::vector<int> solution = std::vector<int>();
            for (int j = 0; j < N; j++)
                solution.push_back(h_solutions[N * i + j] - 1); //"-1" to remove the addition made in the kernel to identify a solution is this array
            solutions->push_back(solution);
        }
    }
    free(h_solutions);

    cudaDeviceSynchronize();
}

void calculateAllSolutions(const int N, const bool print)
{
    std::vector<std::vector<int>> solutions = std::vector<std::vector<int>>();
    int num_solutions = 0;
    auto start = std::chrono::system_clock::now();

    calculateSolutions(N, &solutions, &num_solutions);

    auto stop = std::chrono::system_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "N=" << N << " time elapsed: " << time_elapsed.count() / 1000000.0 << "s\n";

    printf("N=%d, solutions=%d\n\n", N, num_solutions);

    if (print)
    {
        std::string text;
        text.resize(N * (N + 1) + 1); // we know exactly how many characters we'll need: one for each place at the board, and N newlines (at the end of each row). And one more newline to differentiate from other solutions
        text.back() = '\n'; // add extra line at the end
        for (const auto& solution : solutions)
        {
            for (int i = 0; i < N; ++i)
            {
                auto queenAtRow = solution[i];
                for (int j = 0; j < N; ++j)
                    text[i * (N + 1) + j] = queenAtRow == j ? 'X' : '.';
                text[i * (N + 1) + N] = '\n';
            }
            std::cout << text << "\n";
        }
    }
}

__device__ int getGlobalIdx_3D_3D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__global__ void print_dims() {
    printf("%d t=(%d, %d, %d) b=(%d, %d, %d) bd=(%d, %d, %d) gd(%d, %d, %d)\n", getGlobalIdx_3D_3D(), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char** argv)
{
    //gpuErrchk(cudaSetDevice(0));

    for (int N = 4; N <= N_MAX; ++N)
        calculateAllSolutions(N, true);
    
    /* the following code is from an attempted 3D-3D GPU implementation, but I cannot find a way to calculate the total number of threads */
    //double n = std::ceil(pow(N_MAX, N_MAX) / 6);
    //dim3 block = { n, n, n };
    //dim3 grid = { n, n, n };
    //printf("%lf", n);

    /*cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("%d", getSPcores(props));

    print_dims<<<grid, block>>>();
    cudaDeviceSynchronize();*/
}