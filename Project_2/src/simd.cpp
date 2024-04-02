//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// SIMD + Reordering Matrix Multiplication
//

#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

const size_t TILE_SIZE = 128; //128

Matrix matrix_multiply_simd(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N); // Initialize all elements to 0

    int aTile[TILE_SIZE][TILE_SIZE] = {0};
    int bTile[TILE_SIZE][TILE_SIZE] = {0};
    int cTile[TILE_SIZE][TILE_SIZE] = {0}; // Additional tile for storing intermediate results

    for (size_t i = 0; i < M; i += TILE_SIZE) {
        for (size_t j = 0; j < N; j += TILE_SIZE) {
            // Initialize cTile to zeros
            for (size_t x = 0; x < TILE_SIZE; ++x) {
                for (size_t y = 0; y < TILE_SIZE; ++y) {
                    cTile[x][y] = 0;
                }
            }

            for (size_t k = 0; k < K; k += TILE_SIZE) {
                // Load matrix1 tile into aTile
                for (size_t ii = i, x = 0; ii < std::min(i + TILE_SIZE, M); ++ii, ++x) {
                    for (size_t kk = k, y = 0; kk < std::min(k + TILE_SIZE, K); ++kk, ++y) {
                        aTile[x][y] = matrix1[ii][kk];
                    }
                }

                // Load matrix2 tile into bTile
                for (size_t kk = k, x = 0; kk < std::min(k + TILE_SIZE, K); ++kk, ++x) {
                    for (size_t jj = j, y = 0; jj < std::min(j + TILE_SIZE, N); ++jj, ++y) {
                        bTile[x][y] = matrix2[kk][jj];
                    }
                }

                // Multiply the tiles with AVX and accumulate in cTile
                for (size_t x = 0; x < TILE_SIZE; ++x) {
                    for (size_t y = 0; y < TILE_SIZE; ++y) {
                        for (size_t z = 0; z < TILE_SIZE; z += 8) {
                            if (z + 8 <= TILE_SIZE) { // Full SIMD width
                                __m256i aVec = _mm256_set1_epi32(aTile[x][y]);
                                __m256i bVec = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&bTile[y][z]));
                                __m256i resVec = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&cTile[x][z]));

                                __m256i mulVec = _mm256_mullo_epi32(aVec, bVec);
                                resVec = _mm256_add_epi32(resVec, mulVec);

                                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&cTile[x][z]), resVec);
                            } else { // Handle the tail elements
                                for (size_t tail = 0; tail < TILE_SIZE && z + tail < TILE_SIZE; ++tail) {
                                    cTile[x][z + tail] += aTile[x][y] * bTile[y][z + tail];
                                }
                            }
                        }
                    }
                }
            }

            // Copy the results from cTile to the result matrix
            for (size_t x = 0; x < TILE_SIZE && i + x < M; ++x) {
                for (size_t y = 0; y < TILE_SIZE && j + y < N; ++y) {
                    result[i + x][j + y] = cTile[x][y];
                }
            }
        }
    }
    return result;
}



int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_simd(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}