// Headers
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

/*
*   The below N-Queens chessboard formulation is as follows:
*       We know that at a single row, there can only be 1 Queen
*       The state of the chessboard with regards to the Queens' positions then just needs to store N numbers:
*           for each row, store **the column that the queen is located** (which is an integer between 0 and N-1)
*   The algorithm used here works as follows:
*       Create an empty chessboard
*       Try placing a queen at each column in the first row
*       After each such placement, test the state of the chessboard. If it's still valid, then
*           Try placing a queen at each column in the SECOND row (the first row already stores a queen placement at a column there)
*           After each such placement in the second row, test the state of the chessboard. If it's still valid, then
*               Try placing a queen at each column in the THIRD row (the first and second rows already store a queen placement at columns there)
*               ...
*
*    This algorithm is recursive: It applies the same logic again and again, while modifying the internal state.
*    GPUs and parallelism DO NOT WORK WELL WITH RECURSION. So, you need to come up with a solution that achieves the same results WITHOUT RECURSION, so that you can then convert it to OpenMP and GPU
*    Feel free to use existing resources (e.g. how to remove recursion), but REFERENCE EVERYTHING YOU USE, but DON'T COPY-PASTE ANY SOLUTION FROM ANY OBSCURE WEBSITES.
*/

// check if the chessboard is valid so far, for row in [0,lastPlacedRow]
bool boardIsValid(const std::vector<int> gameBoard)
{
    const int N = gameBoard.size();

    // the board needs to be sorted to check that each column is unique
    std::vector<int> gameBoardSorted = gameBoard;
    std::sort(gameBoardSorted.begin(), gameBoardSorted.end());
    if (std::adjacent_find(gameBoardSorted.begin(), gameBoardSorted.end()) != gameBoardSorted.end()) {// same column, fail!
        //std::cout << "returned" << std::endl;
        return false;
    }

    // Check against other queens
    for (int i = 0; i < N; i++) {
        int j = 1;
        while (j < N) {
            if (i == j) {
                j++;
                continue;
            }
            //std::cout << i << " " << j << " +=" << gameBoard[i] + (j) << " -=" << gameBoard[i] - (j) << std::endl;
            if (i - j >= 0) {
                if (gameBoard[i - j] == gameBoard[i] + j || gameBoard[i - j] == gameBoard[i] - j) {
                    //std::cout << "- false" << std::endl;
                    return false;
                }
            }
            if (i + j < N) {
                if (gameBoard[i + j] == gameBoard[i] + j || gameBoard[i + j] == gameBoard[i] - j) {
                    //std::cout << "+ false" << std::endl;
                    return false;
                }
            }
            j++;
        }
    }
    //std::cout << "true" << std::endl;
    return true;
}

void calculateSolutions(std::vector<int>& gameBoard, int N, std::vector<std::vector<int>>& solutions)
{
    int writeToRow = 0;
    for (;;) {
        if (writeToRow < 0 || writeToRow >= N)
            break;
        if (gameBoard[writeToRow] == N - 1)
            writeToRow--;
        else {
            gameBoard[writeToRow]++;
            while (writeToRow < N - 1) {
                writeToRow++;
                gameBoard[writeToRow] = 0;
            }
        }
        /*std::string text;
        text.resize(N * (N + 1) + 1); // we know exactly how many characters we'll need: one for each place at the board, and N newlines (at the end of each row). And one more newline to differentiate from other solutions
        text.back() = '\n'; // add extra line at the end
        for (int i = 0; i < N; ++i)
        {
            auto queenAtRow = gameBoard[i];
            for (int j = 0; j < N; ++j)
                text[i * (N + 1) + j] = queenAtRow == j ? 'X' : '.';
            text[i * (N + 1) + N] = '\n';
        }
        std::cout << text << "\n";*/

        if (boardIsValid(gameBoard))
            solutions.push_back(gameBoard);
    }
}

// Calculate all solutions given the size of the chessboard
void calculateAllSolutions(int N, bool print)
{
    std::vector<std::vector<int>> solutions;
    std::vector<int> gameBoard(N, 0);
    calculateSolutions(gameBoard, N, solutions);
    printf("N=%d, solutions=%d\n", N, int(solutions.size()));

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