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

inline bool boardIsValid(const int* gameBoard, const int N)
/* use this function definition when using the vector format of the "gameBoard" in "calculateSolutions()"
bool boardIsValid(const std::vector<int> gameBoard, const int N)*/
{
    /* this OpenMP parallelization only slows the application down, most likely because it has an additional "if(!valid)" check
     * the "if(!valid)" check exists as it's not possible to return from a Parallel For, so this skips future checks if an invalid scenario is encountered
     * but, with this solution, it's still going to have to finish iterating; when i == N
     */
    /*volatile bool valid = true;

#pragma omp parallel for num_threads(std::round(std::thread::hardware_concurrency() / 2)) schedule(static) shared(valid)
    for (int i = 0; i < N; i++) {
        if (!valid)
            continue;

        for (int j = i + 1; j < N; j++) {
            if (!valid)
                continue;

            if (gameBoard[i] - gameBoard[j] == i - j || gameBoard[i] - gameBoard[j] == j - i || gameBoard[i] == gameBoard[j])
                valid = false;
        }
    }

    return valid;*/

    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (gameBoard[i] - gameBoard[j] == i - j || gameBoard[i] - gameBoard[j] == j - i || gameBoard[i] == gameBoard[j])
                return false;
    return true;
}

void calculateSolutions(int N, std::vector<std::vector<int>>& solutions)
{
    int O = pow(N, N);
    
    /* the following two lines are for an array-based solution, alternative to the "solutions" vector, as it may increase OpenMP's performance
    int** solutions_array = nullptr;
    int num_solutions = 0;*/

/* use this commented out preprocessing argument when using the parallelized "boardIsValid" solution
#pragma omp parallel for num_threads(std::round(std::thread::hardware_concurrency() / 2)) schedule(dynamic) */
#pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(dynamic) // dynamic scheduling is best as we don't know whether a permutation is going to be valid or not and, therefore, utilise the "boardIsValid" check during task time
    for (int i = 0; i < O; i++) {
        int* gameBoard = (int*)malloc(sizeof(int) * N); // OpenMP's performance improves drastically when using an array instead of a vector
        //std::vector<int> gameBoard(N, 0); // vector implementation of "gameBoard" - always runs slower than an array

        int column = i;
        for (int j = 0; j < N; j++) {
            gameBoard[j] = column % N;
            column /= N;
        }

        if (boardIsValid(gameBoard, N)) {
#pragma omp critical // when using a solution other than the dynamically allocated "int** solutions_array", use this line and one of the two below
            solutions.push_back(std::vector<int>(gameBoard, gameBoard + sizeof gameBoard / sizeof gameBoard[0])); // if "gameBoard" is an array, use this line
            //solutions.push_back(gameBoard); // if "gameBoard" is a vector, use this line
            
            /* make sure to use the following three lines when using the dynamically allocated "int** solutions_array"
            num_solutions++;
            solutions_array = (int**)realloc(solutions_array, sizeof(int*) * num_solutions);
            solutions_array[num_solutions - 1] = gameBoard;*/
        }
        /* make sure to use this "free()" when using the dynamically allocated "int* gameBoard" but not when using the "int** solutions_array" */
        free(gameBoard);
    }

    /* make sure to use the following four lines when using the dynamically allocated "int** solutions_array"
    for (int i = 0; i < num_solutions; i++) {
        solutions.push_back(std::vector<int>(solutions_array[i], solutions_array[i] + sizeof solutions_array[i] / sizeof solutions_array[i][0]));
        free(solutions_array[i]);
    }
    free(solutions_array);*/
}

// Calculate all solutions given the size of the chessboard
void calculateAllSolutions(int N, bool print)
{
    std::vector<std::vector<int>> solutions;

    auto start = omp_get_wtime();
    calculateSolutions(N, solutions);
    auto stop = omp_get_wtime();

    auto time_elapsed = stop - start;
    std::cout << "N=" << N << " time elapsed: " << time_elapsed << "s\n";
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