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

#include "gpuErrchk.h"

#define N_MAX 12 //max board N*N; this is the max board's dimensions' value to evaluate, so the program will evaluate solutions to each N*N board up to this value (i.e. 4 <= N <= N_MAX)

////////////
//if the program is weilding incorrect results then these constants may need to be modified to match your GPU specification

#define TPB 512 //this constant is used to allocate 2D GPU "Threads Per Block" (if you'd like to compare it to the current 3D-3D implementation)

#define GRID_X 1024
#define GRID_Y 14
#define GRID_Z 2
#define BLOCK_X 16
#define BLOCK_Y 14
#define BLOCK_Z 2
#define N_THREADS (const long long int)(GRID_X * GRID_Y * GRID_Z)
////////////

__device__ inline int getGlobalIdx_3D_3D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

/* use this kernel for an integer-based "gameBoard" gameboard implementation 
__device__ bool boardIsValidSoFar(const int lastPlacedRow, const long long int gameBoard, const int N, const int reduction)
{
    int lastPlacedColumn = gameBoard % reduction;

    for (int row = 0; row < lastPlacedRow; ++row)
    {
        int column = (gameBoard / (long long int)pow(reduction, (lastPlacedRow) - row)) % reduction;
        
        if (column == lastPlacedColumn)
            return false;
        const auto col1 = lastPlacedColumn - (lastPlacedRow - row);
        const auto col2 = lastPlacedColumn + (lastPlacedRow - row);
        if (column == col1 || column == col2)
            return false;
    }
    return true;
}

__global__ void permutationGenAndEval(const int N, const long long int O, const long long int offset, int* d_solutions, int* d_num_solutions, const int reduction) {
    //long long int column = (long long int)getGlobalIdx_3D_3D() + offset; //use this line for the 3D-3D implementation
    long long int column = (long long int)(threadIdx.x + blockIdx.x * blockDim.x) + offset; //use this line for the 2D implementation
    if (column >= O)
        return;

    bool valid = true;
    long long int gameBoard = 0;
    for (int i = 0; i < N; i++) {
        gameBoard *= reduction;
        gameBoard += column % N;

        if (!boardIsValidSoFar(i, gameBoard, N, reduction)) {
            valid = false;
            break;
        }

        column /= N;
    }

    if (valid) { //I tried combining this block of code with the "if" in the "for" loop above it, but this ended up being slower by about 4% (on average)
        const int index = atomicAdd(d_num_solutions, 1);
        d_solutions[index] = gameBoard;
    }
}*/

/* use this kernel for an array-based "gameBoard" implementation */
__device__ bool boardIsValidSoFar(int lastPlacedRow, const int* gameBoard, const int N)
{
    int lastPlacedColumn = gameBoard[lastPlacedRow];

    for (int row = 0; row < lastPlacedRow; ++row)
    {
        if (gameBoard[row] == lastPlacedColumn)
            return false;
        const auto col1 = lastPlacedColumn - (lastPlacedRow - row);
        const auto col2 = lastPlacedColumn + (lastPlacedRow - row);
        if (gameBoard[row] == col1 || gameBoard[row] == col2)
            return false;
    }
    return true;
}

__global__ void permutationGenAndEval(const int N, const long long int O, const long long int offset, int* d_solutions, int* d_num_solutions) {
    //long long int column = (long long int)getGlobalIdx_3D_3D() + offset; //use this line for the 3D-3D implementation
    long long int column = (long long int)(threadIdx.x + blockIdx.x * blockDim.x) + offset; //use this line for the 2D implementation
    if (column >= O)
        return;

    bool valid = true;
    int gameBoard[N_MAX];
    for (int i = 0; i < N; i++) {
        gameBoard[i] = column % N;

        if (!boardIsValidSoFar(i, gameBoard, N)) {
            valid = false;
            break;
        }

        column /= N;
    }

    if (valid) { //I tried combining this block of code with the "if" in the "for" loop above it, but this ended up being slower by about 4% (on average)
        const int index = atomicAdd(d_num_solutions, 1);
        for (int i = 0; i < N; i++)
            d_solutions[N * index + i] = gameBoard[i]; //"+1" so that we can tell later which indexes of "d_solutions" are empty using 0
    }
}

void initialiseDevice(const int N, std::vector<std::vector<int>>* solutions, int* h_num_solutions)
{
    *h_num_solutions = 0;
    int* d_solutions = nullptr;
    int* d_num_solutions = nullptr;

    const long long int O = powl(N, N);

    size_t solutions_mem = pow(N, 5) * sizeof(int*); //N^5 is an estimation of the amount of solutions for size N (^5 because N_MAX^4 (12^4) is enough to hold all the solutions for a 12x12 board and to store N columns for that board that would make it N^5)
    cudaMalloc((void**)&d_solutions, solutions_mem);
    cudaMalloc((void**)&d_num_solutions, sizeof(int));

    cudaMemcpy(d_num_solutions, h_num_solutions, sizeof(int), cudaMemcpyHostToDevice);
    
    int id_offsets = 1; //initialise as 1 so that the kernel is executed at least once
    
    /* use the following four lines with the 3D-3D implementation 
    if (O > N_THREADS)
        id_offsets = std::ceil((double)O / N_THREADS); //calculate the amount of thread ID offsets needed (round up to account for the remainder offset)
    dim3 block = {BLOCK_X, BLOCK_Y, BLOCK_Z};
    dim3 grid = { GRID_X / BLOCK_X, GRID_Y / BLOCK_Y, GRID_Z / BLOCK_Z };*/

    /* use these two lines with the 2D kernel implementation */
    int grid = TPB * 2;
    int block = TPB;
    if (O > grid * block)
        id_offsets = std::ceil((double)O / (grid * block));

    /* this variable is used for the integer - based "gameBoard" implementation 
    const int reduction = N >= 10 ? 100 : 10;*/

    for (long long int i = 0; i < id_offsets; i++) {
        /* use either of these kernel invocations with the integer-based "gameBoard" implementation
        //permutationGenAndEval<<<grid, block>>>(N, O, N_THREADS * i, d_solutions, d_num_solutions); //use this kernel invocation for the 3D-3D implementation
        permutationGenAndEval<<<grid, block>>>(N, O, (long long int)grid * block * i, d_solutions, d_num_solutions); //use this kernel invocation for the 2D implementation*/

        /* use either of these kernel invocations with the array-based "gameBoard" implmentation */
        //permutationGenAndEval<<<grid, block>>>(N, O, N_THREADS * i, d_solutions, d_num_solutions); //use this kernel invocation for the 3D-3D implementation
        permutationGenAndEval<<<grid, block>>>(N, O, (long long int)grid * block * i, d_solutions, d_num_solutions); //use this kernel invocation for the 2D implementation
        
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_num_solutions, d_num_solutions, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_num_solutions);
    
    int* h_solutions = (int*)malloc(solutions_mem);
    cudaMemcpy(h_solutions, d_solutions, solutions_mem, cudaMemcpyDeviceToHost);
    cudaFree(d_solutions);

    /* use this for loop with the integer-based "gameBoard" implementation 
    for (int i = 0; i < *h_num_solutions; i++) {
        std::vector<int> solution = std::vector<int>();
        for (int j = 0; j < N; j++) {
            solution.push_back((h_solutions[i] % reduction));
            h_solutions[i] /= reduction;
        }
        solutions->push_back(solution);
    }*/

    /* use this for loop with the array-based "gameBoard" implementation */
    for (int i = 0; i < *h_num_solutions; i++) {
        std::vector<int> solution = std::vector<int>();
        for (int j = 0; j < N; j++)
            solution.push_back(h_solutions[N * i + j]);
        solutions->push_back(solution);
    }

    free(h_solutions);
}

void calculateAllSolutions(const int N, const bool print)
{
    std::vector<std::vector<int>> solutions = std::vector<std::vector<int>>();
    int num_solutions = 0;

    //time the entire device uptime instead of only the permutations generation in order to account for the device memory management time as well
    auto start = std::chrono::system_clock::now();
    initialiseDevice(N, &solutions, &num_solutions);
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
    gpuErrchk(cudaSetDevice(0));

    for (int N = 4; N <= N_MAX; ++N)
        calculateAllSolutions(N, false);
}