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

#define N_MAX 12

/* use this commented out function declaration when using "gameBoard" as a vector
inline bool boardIsValidSoFar(int lastPlacedRow, const std::vector<int>& gameBoard, const int N)*/
inline bool boardIsValidSoFar(int lastPlacedRow, const int* gameBoard, const int N)
{
    int lastPlacedColumn = gameBoard[lastPlacedRow];

    /* use this boolean when the below for loop is a Parallel For
    volatile bool valid = true;*/

//#pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(dynamic) shared(valid)
    for (int row = 0; row < lastPlacedRow; ++row)
    {
        /* use this condition when this for is parallel
        if (!valid)
            continue;*/

        if (gameBoard[row] == lastPlacedColumn) // same column, fail!
            /* use this, and the following, returns when this for is not parallel (other wise use the following uses of the "valid" variable */
            return false;
            //valid = false;
        // check the 2 diagonals
        const auto col1 = lastPlacedColumn - (lastPlacedRow - row);
        const auto col2 = lastPlacedColumn + (lastPlacedRow - row);
        if (gameBoard[row] == col1 || gameBoard[row] == col2)
            return false;
            //valid = false;
    }
    return true;
    //return valid;
}

void calculateSolutions(int N, std::vector<std::vector<int>>& solutions)
{
    const long long int O = powl(N, N);

    /* the following two lines are for an array-based solution, alternative to the "solutions" vector */
    int* solutions_array = (int*)malloc(pow(N, 5) * sizeof(int)); //N^5 is an estimation of the amount of solutions for size N (^5 because N_MAX^4 (12^4) is enough to hold all the solutions for a 12x12 board and to store N columns for that board that would make it N^5)
    std::atomic<int> num_solutions = 0;

#pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(dynamic) // dynamic scheduling is best as we don't know whether a permutation is going to be valid or not and, therefore, utilise the full "boardIsValidSoFar" check during task time
    for (long long int i = 0; i < O; i++) {

        bool valid = true;
        int gameBoard[N_MAX]; // OpenMP's performance improves drastically when using an array instead of a vector
        //std::vector<int> gameBoard(N, 0); // vector implementation of "gameBoard" - always runs slower than an array

        long long int column = i;
        for (int j = 0; j < N; j++) {
            gameBoard[j] = column % N;

            if (!boardIsValidSoFar(j, gameBoard, N)) {
                valid = false;
                break;
            }

            column /= N;
        }

        if (valid) {
/* when using a solution other than the dynamically allocated "int* solutions_array", use two of the following three lines
#pragma omp critical 
            solutions.push_back(std::vector<int>(gameBoard, gameBoard + sizeof gameBoard / sizeof gameBoard[0])); // if "gameBoard" is an array, use this line
            //solutions.push_back(gameBoard); // if "gameBoard" is a vector, use this line*/

            /* make sure to use the following three lines when using the dynamically allocated "int** solutions_array" */
            for (int j = 0; j < N; j++)
                solutions_array[N * num_solutions + j] = gameBoard[j];
            num_solutions++;
        }
    }

    /* make sure to use the following four lines when using the dynamically allocated "int** solutions_array" */
    for (int i = 0; i < num_solutions; i++) {
        std::vector<int> solution = std::vector<int>();
        for (int j = 0; j < N; j++)
            solution.push_back(solutions_array[N * i + j]);
        solutions.push_back(solution);
    }
    free(solutions_array);
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
    for (int N = 4; N <= N_MAX; ++N)
        calculateAllSolutions(N, false);
}