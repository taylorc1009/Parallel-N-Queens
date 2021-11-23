// Headers
#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <stack>

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
bool boardIsValidSoFar(int lastPlacedRow, const std::vector<int>& gameBoard)
{
    const auto N = gameBoard.size();
    int lastPlacedColumn = gameBoard[lastPlacedRow];

    // Check against other queens
    for (int row = 0; row < lastPlacedRow; ++row)
    {
        if (gameBoard[row] == lastPlacedColumn) // same column, fail!
            return false;
        // check the 2 diagonals
        const auto col1 = lastPlacedColumn - (lastPlacedRow - row);
        const auto col2 = lastPlacedColumn + (lastPlacedRow - row);
        if (gameBoard[row] == col1 || gameBoard[row] == col2)
            return false;
    }
    return true;
}

// A recursive function to calculate solutions
void calculateSolutions(std::vector<int>& gameBoard, int N, std::vector<std::vector<int>>& solutions)
{
    std::stack<std::pair<int, int>> rowWriteHistory = std::stack<std::pair<int, int>>();
    int writeToRow = 0;

    //for (gameBoard[0] = 0; gameBoard[0] < N; gameBoard[0]++) //for each column
    //{
        for (int i = 0; i < N; i++)
        {
            gameBoard[writeToRow] = i;
            
            /*std::string text;
            text.resize(N * (N + 1) + 1); // we know exactly how many characters we'll need: one for each place at the board, and N newlines (at the end of each row). And one more newline to differentiate from other solutions
            text.back() = '\n'; // add extra line at the end
            for (int k = 0; k < N; ++k) {
                auto queenAtRow = gameBoard[k];
                for (int j = 0; j < N; ++j)
                    text[k * (N + 1) + j] = queenAtRow == j ? 'X' : '.';
                text[k * (N + 1) + N] = '\n';
            }
            std::cout << text;// << "\n";
            std::cout << "wTR " << writeToRow << ", j " << i << std::endl << std:: endl << std::endl;*/

            if (boardIsValidSoFar(writeToRow, gameBoard))
            {
                //std::cout << "valid" << std::endl;
                if (writeToRow == N - 1) {
                    solutions.push_back(gameBoard);
                    //std::cout << "stored" << std::endl;
                }
                else {
                    rowWriteHistory.push(std::make_pair(writeToRow, i));
                    //std::cout << "stacked: " << writeToRow << ", " << i << " (size: " << rowWriteHistory.size() << ')' << std::endl;
                    writeToRow++;
                    i = -1;
                }
            }
            //else if (i == N - 1) {//&& !rowWriteHistory.empty()) {
            if (i == N - 1) {//&& !rowWriteHistory.empty()) {
                do {
                    if (!rowWriteHistory.empty()) {
                        std::pair<int, int> tempPair = rowWriteHistory.top();
                        //std::cout << rowWriteHistory.size() << ": " << writeToRow << ", " << i << '\n';
                        //std::cout << "rWH: " << rowWriteHistory.top().first << ", " << rowWriteHistory.top().second << std::endl;
                        writeToRow = tempPair.first;
                        i = tempPair.second;
                        rowWriteHistory.pop();
                    }
                    else
                        break;
                } while (i == N - 1);
            }
        }
    //}
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