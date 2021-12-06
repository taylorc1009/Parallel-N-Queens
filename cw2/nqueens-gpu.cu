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
#include <cmath>

//#include "gpuErrchk.h"

#define N_MAX 12

#define GRID_X 1024
#define GRID_Y 14
#define GRID_Z 2
#define BLOCK_X 16
#define BLOCK_Y 14
#define BLOCK_Z 2
#define N_THREADS (const long long int)(GRID_X * GRID_Y * GRID_Z)

__device__ int getGlobalIdx_3D_3D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__device__ bool boardIsValid(const int* gameBoard, const int N)
{
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (gameBoard[i] - gameBoard[j] == i - j || gameBoard[i] - gameBoard[j] == j - i || gameBoard[i] == gameBoard[j])
                return false;
    return true;
}

__global__ void getPermutations(const int N, const long long int O, const long long int offset, int* d_solutions, int* d_num_solutions) {
    long long column = (long long int)getGlobalIdx_3D_3D() + offset;
    if (column >= O)
        return;

    int gameBoard[N_MAX];
    for (int i = 0; i < N; i++) {
        gameBoard[i] = column % N;
        column /= N;
    }

    if (boardIsValid(gameBoard, N)) {
        const int index = atomicAdd(d_num_solutions, 1);
        for (int i = 0; i < N; i++)
            d_solutions[N * index + i] = gameBoard[i] + 1; //"+1" so that we can tell later which indexes of "d_solutions" are empty using 0
    }

    __syncthreads();
}

void calculateSolutions(const int N, std::vector<std::vector<int>>* solutions, int* h_num_solutions)
{
    *h_num_solutions = 0;
    int* d_solutions = nullptr;
    int* d_num_solutions = nullptr;

    const long long int O = powl(N, N);

    size_t solutions_mem = pow(N, 5) * sizeof(int*); // N^5 is an estimation of the amount of solutions for size N (^5 because N_MAX^4 (12^4) is enough to hold all the solutions for a 12x12 board and to store N columns for that board that would make it N^5)
    cudaMalloc((void**)&d_solutions, solutions_mem);
    cudaMalloc((void**)&d_num_solutions, sizeof(int));

    cudaMemcpy(d_num_solutions, h_num_solutions, sizeof(int), cudaMemcpyHostToDevice);
    
    int offsets = 1; //initialise as 1 so that the kernel is executed at least once
    if (O > N_THREADS)
        offsets = std::ceil(O / N_THREADS);

    dim3 block = { BLOCK_X, BLOCK_Y, BLOCK_Z };
    dim3 grid = { GRID_X / BLOCK_X, GRID_Y / BLOCK_Y, GRID_Z / BLOCK_Z };
    for (long long int i = 0; i < offsets; i++) {
        getPermutations<<<grid, block>>>(N, O, (long long int)(N_THREADS * i), d_solutions, d_num_solutions);
        //if (N >= 10) printf("%d, %d, %lld\n", offsets, i, (long long int)(N_THREADS * i));
        cudaDeviceSynchronize();
    }

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

int main(int argc, char** argv)
{
    //gpuErrchk(cudaSetDevice(0));

    for (int N = 4; N <= N_MAX; ++N)
        calculateAllSolutions(N, false);
}