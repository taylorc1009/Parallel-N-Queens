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

bool boardIsValid(const std::vector<int> gameBoard)
{
    const int N = gameBoard.size();

    std::vector<int> gameBoardSorted = gameBoard;
    std::sort(gameBoardSorted.begin(), gameBoardSorted.end());
    if (std::adjacent_find(gameBoardSorted.begin(), gameBoardSorted.end()) != gameBoardSorted.end())// same column, fail!
        return false;

    for (int i = 0; i < N; i++) {
        int j = 1;
        while (j < N) {
            if (i == j) {
                j++;
                continue;
            }
            if (i - j >= 0)
                if (gameBoard[i - j] == gameBoard[i] + j || gameBoard[i - j] == gameBoard[i] - j)
                    return false;
            if (i + j < N)
                if (gameBoard[i + j] == gameBoard[i] + j || gameBoard[i + j] == gameBoard[i] - j)
                    return false;
            j++;
        }
    }
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