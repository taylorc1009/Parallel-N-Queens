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

//#include "gpuErrchk.h"

#define N_MAX 12

#define GRID_X 1024
#define GRID_Y 14
#define GRID_Z 2
#define DATA_SIZE (GRID_X * GRID_Y * GRID_Z)
#define BLOCK_X 16
#define BLOCK_Y 14
#define BLOCK_Z 2

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

__host__ __device__ int* boardMalloc(const int* gameBoard, size_t size) {
    int* dynamicBoard = nullptr;
    unsigned int dynamicBoard_size = 0;
    cudaMalloc((void**)&dynamicBoard, size);
    for (int i = 0; i < N_MAX; i++) {
        if (gameBoard[i] != -1) {
            dynamicBoard[dynamicBoard_size] = gameBoard[i];
            dynamicBoard_size++;
            //printf("dB[%d]=%d", dynamicBoard_size, gameBoard[i]);
        }
    }
    return dynamicBoard;
}

__global__ void getPermutations(const int N, const int O, int** d_permutations, int* d_num_solutions) {
    int column = threadIdx.x + blockIdx.x * blockDim.x;
    if (column >= O)
        return;
    //printf("%d=%d, t=(%d, %d, %d) b=(%d, %d, %d) bd=(%d, %d, %d) gd(%d, %d, %d)\n", N, column, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);

    int gameBoard[N_MAX];
    for (int i = 0; i < N_MAX; i++)
        gameBoard[i] = -1;

    for (int j = 0; j < N; j++) {
        //printf("%d %d %d %d\n", N, threadIdx.x + blockIdx.x * blockDim.x, column, column % N);
        gameBoard[j] = column % N;
        column /= N;
    }

    __syncthreads();

    if (boardIsValid(gameBoard, N)) {
        int index = atomicAdd(d_num_solutions, 1);
        d_permutations[index] = boardMalloc(gameBoard, sizeof(int) * N);
        //printf("%d\n", d_permutations[index][0]);
    }

    __syncthreads();
}

void calculateSolutions(const int N, int** h_solutions, int& h_num_solutions)
{
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(unsigned(N) / BLOCK_X, unsigned(N) / BLOCK_Y, unsigned(N) / BLOCK_Z);

    h_num_solutions = 0;
    int** d_solutions = nullptr;
    int* d_num_solutions = nullptr;

    int O = pow(N, N);

    cudaMalloc((void***)&d_solutions, O * sizeof(int*));
    cudaMalloc((void**)&d_num_solutions, sizeof(int));

    cudaMemcpy(d_num_solutions, &h_num_solutions, sizeof(int), cudaMemcpyHostToDevice);

    printf("block=(%d, %d, %d) grid=(%d, %d, %d)\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
    
    getPermutations<<<(O + 512 - 1) / 512, 512>>>(N, O, d_solutions, d_num_solutions);

    cudaMemcpy(&h_num_solutions, d_num_solutions, sizeof(int), cudaMemcpyDeviceToHost);
    h_solutions = (int**)malloc(sizeof(int*) * h_num_solutions);
    //for (int i = 0; i < h_num_solutions; i++) {
        //if (d_solutions[i])
            cudaMemcpy(&h_solutions, &d_solutions, O * sizeof(int*), cudaMemcpyDeviceToHost);
        cudaFree(d_solutions);
    //}
    cudaFree(d_num_solutions);
    //cudaFree(d_solutions);

    cudaDeviceSynchronize();
}

// Calculate all solutions given the size of the chessboard
void calculateAllSolutions(const int N, const bool print)
{
    int** solutions = nullptr;
    int num_solutions = 0;
    auto start = std::chrono::system_clock::now();

    calculateSolutions(N, solutions, num_solutions);

    auto stop = std::chrono::system_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "N=" << N << " time elapsed: " << time_elapsed.count() / 1000.0 << "s\n";

    printf("N=%d, solutions=%d\n\n", N, num_solutions);

    if (print)
    {
        std::string text;
        text.resize(N * (N + 1) + 1); // we know exactly how many characters we'll need: one for each place at the board, and N newlines (at the end of each row). And one more newline to differentiate from other solutions
        text.back() = '\n'; // add extra line at the end
        for (int i = 0; i < num_solutions; i++)
        {
            for (int j = 0; j < N; j++)
            {
                auto queenAtRow = solutions[i][j];
                for (int j = 0; j < N; ++j)
                    text[i * (N + 1) + j] = queenAtRow == j ? 'X' : '.';
                text[i * (N + 1) + N] = '\n';
            }
            std::cout << text << "\n";
        }
    }

    if (solutions != nullptr)
        free(solutions);
}


int main(int argc, char** argv)
{
    //gpuErrchk(cudaSetDevice(0));

    for (int N = 4; N <= N_MAX; ++N)
        calculateAllSolutions(N, false);
}