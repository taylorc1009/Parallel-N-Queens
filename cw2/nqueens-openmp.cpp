#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <stack>
#include <thread>
#include <omp.h>
#include <algorithm>

bool boardIsValid(const int* gameBoard, const int N)
{
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (gameBoard[i] - gameBoard[j] == i - j || gameBoard[i] - gameBoard[j] == j - i || gameBoard[i] == gameBoard[j])
                return false;
    return true;
}

void calculateSolutions(int N, std::vector<std::vector<int>>& solutions)
{
    int O = pow(N, N);
    /*int** solutions_array = nullptr;
    int num_solutions = 0;*/

    auto start = std::chrono::system_clock::now();

#pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(static, N)
    for (int i = 0; i < O; i++) {
        int* gameBoard = (int*)malloc(sizeof(int) * N); // OpenMP's performance improves drastically when using arrays instead of vectors

        int column = i;
        for (int j = 0; j < N; j++) {
            gameBoard[j] = column % N;
            column /= N;
        }

        if (boardIsValid(gameBoard, N)) {
#pragma omp critical
            solutions.push_back(std::vector<int>(gameBoard, gameBoard + sizeof gameBoard / sizeof gameBoard[0]));
            /*num_solutions++;
            solutions_array = (int**)realloc(solutions_array, sizeof(int*) * num_solutions);
            solutions_array[num_solutions - 1] = gameBoard;*/
        }
    }

    auto stop = std::chrono::system_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "N=" << N << " time elapsed: " << time_elapsed.count() / 1000.0 << "s\n";

    /*for (int i = 0; i < num_solutions; i++) {
        solutions.push_back(std::vector<int>(solutions_array[i], solutions_array[i] + sizeof solutions_array[i] / sizeof solutions_array[i][0]));
        free(solutions_array[i]);
    }
    free(solutions_array);*/
}

// Calculate all solutions given the size of the chessboard
void calculateAllSolutions(int N, bool print)
{
    std::vector<std::vector<int>> solutions;
    calculateSolutions(N, solutions);
    printf("N=%d, solutions=%d\n\n", N, int(solutions.size()));

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
    for (int N = 4; N < 13; ++N)
        calculateAllSolutions(N, false);
}