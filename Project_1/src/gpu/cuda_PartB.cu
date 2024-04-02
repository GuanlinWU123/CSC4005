#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

__constant__ double d_filter[FILTER_SIZE][FILTER_SIZE];

// __global__ void applyFilter(const unsigned char* input, unsigned char* output,
//                             int width, int height, int num_channels)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < width * height * num_channels)
//     {
//         int pixel_idx = idx / num_channels;
//         int channel_idx = idx % num_channels;
//         int row = pixel_idx / width;
//         int col = pixel_idx % width;

//         double sum = 0.0;
//         for (int i = -1; i <= 1; i++)
//         {
//             for (int j = -1; j <= 1; j++)
//             {
//                 int row_idx = row + i;
//                 int col_idx = col + j;
//                 if (row_idx >= 0 && row_idx < height && col_idx >= 0 && col_idx < width)
//                 {
//                     int input_idx = (row_idx * width + col_idx) * num_channels + channel_idx;
//                     sum += static_cast<double>(input[input_idx]) * d_filter[i + 1][j + 1];
//                 }
//             }
//         }
//         output[idx] = static_cast<unsigned char>(std::round(sum));
//     }
// }

__global__ void applyFilter(const unsigned char* input, unsigned char* output,
                            int width, int height, int num_channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height * num_channels)
    {
        int pixel_idx = idx / num_channels;
        int channel_idx = idx % num_channels;
        int row = pixel_idx / width;
        int col = pixel_idx % width;

        double sum = 0.0;

        // Unrolled loop for i = -1, 0, 1 and j = -1, 0, 1
        int row_idx, col_idx, input_idx;

        // i = -1
        row_idx = row - 1;
        col_idx = col - 1;
        if (row_idx >= 0 && row_idx < height && col_idx >= 0 && col_idx < width)
        {
            input_idx = (row_idx * width + col_idx) * num_channels + channel_idx;
            sum += static_cast<double>(input[input_idx]) * d_filter[0][0];
        }

        row_idx = row - 1;
        col_idx = col;
        if (row_idx >= 0 && row_idx < height)
        {
            input_idx = (row_idx * width + col_idx) * num_channels + channel_idx;
            sum += static_cast<double>(input[input_idx]) * d_filter[0][1];
        }

        row_idx = row - 1;
        col_idx = col + 1;
        if (row_idx >= 0 && row_idx < height && col_idx >= 0 && col_idx < width)
        {
            input_idx = (row_idx * width + col_idx) * num_channels + channel_idx;
            sum += static_cast<double>(input[input_idx]) * d_filter[0][2];
        }

        // i = 0
        row_idx = row;
        col_idx = col - 1;
        if (col_idx >= 0 && col_idx < width)
        {
            input_idx = (row_idx * width + col_idx) * num_channels + channel_idx;
            sum += static_cast<double>(input[input_idx]) * d_filter[1][0];
        }

        row_idx = row;
        col_idx = col;
        input_idx = (row_idx * width + col_idx) * num_channels + channel_idx;
        sum += static_cast<double>(input[input_idx]) * d_filter[1][1];

        row_idx = row;
        col_idx = col + 1;
        if (col_idx >= 0 && col_idx < width)
        {
            input_idx = (row_idx * width + col_idx) * num_channels + channel_idx;
            sum += static_cast<double>(input[input_idx]) * d_filter[1][2];
        }

        // i = 1
        row_idx = row + 1;
        col_idx = col - 1;
        if (row_idx >= 0 && row_idx < height && col_idx >= 0 && col_idx < width)
        {
            input_idx = (row_idx * width + col_idx) * num_channels + channel_idx;
            sum += static_cast<double>(input[input_idx]) * d_filter[2][0];
        }

        row_idx = row + 1;
        col_idx = col;
        if (row_idx >= 0 && row_idx < height)
        {
            input_idx = (row_idx * width + col_idx) * num_channels + channel_idx;
            sum += static_cast<double>(input[input_idx]) * d_filter[2][1];
        }

        row_idx = row + 1;
        col_idx = col + 1;
        if (row_idx >= 0 && row_idx < height && col_idx >= 0 && col_idx < width)
        {
            input_idx = (row_idx * width + col_idx) * num_channels + channel_idx;
            sum += static_cast<double>(input[input_idx]) * d_filter[2][2];
        }

        output[idx] = static_cast<unsigned char>(std::round(sum));
    }
}


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    
    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename); // Implement read_from_jpeg

    // Allocate memory on host (CPU)
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char));

    // Copy input data from host to device
    cudaMemcpy(d_input, input_jpeg.buffer, input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Copy the filter from host to constant memory on the device
    cudaMemcpyToSymbol(d_filter, filter, FILTER_SIZE * FILTER_SIZE * sizeof(double));

    // Compute the block and grid dimensions for CUDA
    int blockSize = 512; // Adjust as needed
    int numBlocks = (input_jpeg.width * input_jpeg.height * input_jpeg.num_channels + blockSize - 1) / blockSize;

    // Start GPU timer
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch CUDA kernel to apply the filter
    applyFilter<<<numBlocks, blockSize>>>(d_input, d_output, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels);

    // Stop GPU timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuDuration, start, stop);

    // Copy output data from device to host
    cudaMemcpy(filteredImage, d_output, input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space}; // Implement JPEGMeta

    if (write_to_jpeg(output_jpeg, output_filepath)) // Implement write_to_jpeg
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Cleanup memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input_jpeg.buffer;
    delete[] filteredImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}









