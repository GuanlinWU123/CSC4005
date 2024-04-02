#include <iostream>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

#define MASTER 0
#define TAG_COMPUTE 0

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    
    int numtasks, taskid;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Status status;
    
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        MPI_Finalize();
        return -1;
    }
    
    const char* input_filename = argv[1];
    auto input_jpeg = read_from_jpeg(input_filename);
    
    int rowsPerTask = input_jpeg.height / numtasks;
    int remainingRows = input_jpeg.height % numtasks;
    
    if (taskid == MASTER) {
        std::cout << "Input file from: " << input_filename << "\n";
    }
    
    int startRow = taskid * rowsPerTask + std::min(taskid, remainingRows);
    int endRow = startRow + rowsPerTask - 1;
    if (taskid < remainingRows) endRow++;

    auto filteredImage = new unsigned char[input_jpeg.width * (endRow - startRow + 1) * input_jpeg.num_channels]();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    // for (int height = std::max(1, startRow); height <= std::min(endRow, input_jpeg.height - 2); height++) {
    //     for (int width = 1; width < input_jpeg.width - 1; width++) {
    //         double sum_r = 0, sum_g = 0, sum_b = 0;
    //         for (int i = -1; i <= 1; i++) {
    //             for (int j = -1; j <= 1; j++) {
    //                 int idx = ((height + i) * input_jpeg.width + (width + j)) * input_jpeg.num_channels;
    //                 sum_r += input_jpeg.buffer[idx] * filter[i + 1][j + 1];
    //                 sum_g += input_jpeg.buffer[idx + 1] * filter[i + 1][j + 1];
    //                 sum_b += input_jpeg.buffer[idx + 2] * filter[i + 1][j + 1];
    //             }
    //         }
    //         int idx = ((height - startRow) * input_jpeg.width + width) * input_jpeg.num_channels;
    //         filteredImage[idx] = static_cast<unsigned char>(std::round(sum_r));
    //         filteredImage[idx + 1] = static_cast<unsigned char>(std::round(sum_g));
    //         filteredImage[idx + 2] = static_cast<unsigned char>(std::round(sum_b));
    //     }
    // }

    int minHeight = std::max(1, startRow);
    int maxHeight = std::min(endRow, input_jpeg.height - 2);
    int maxWidth = input_jpeg.width - 1;

    for (int height = minHeight; height <= maxHeight; height++) {
        for (int width = 1; width < maxWidth; width++) {
            double sum_r = 0, sum_g = 0, sum_b = 0;
            unsigned char* base_ptr = &input_jpeg.buffer[(height * input_jpeg.width + width) * input_jpeg.num_channels];
            
            //upper row
            unsigned char* ptr = base_ptr - input_jpeg.width * input_jpeg.num_channels - input_jpeg.num_channels;
            sum_r += *ptr * filter[0][0]; ptr++;
            sum_g += *ptr * filter[0][0]; ptr++;
            sum_b += *ptr * filter[0][0];
            
            ptr = base_ptr - input_jpeg.width * input_jpeg.num_channels;
            sum_r += *ptr * filter[0][1]; ptr++;
            sum_g += *ptr * filter[0][1]; ptr++;
            sum_b += *ptr * filter[0][1];
            
            ptr = base_ptr - input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels;
            sum_r += *ptr * filter[0][2]; ptr++;
            sum_g += *ptr * filter[0][2]; ptr++;
            sum_b += *ptr * filter[0][2];
            
            //middle row
            ptr = base_ptr - input_jpeg.num_channels;
            sum_r += *ptr * filter[1][0]; ptr++;
            sum_g += *ptr * filter[1][0]; ptr++;
            sum_b += *ptr * filter[1][0];
            
            ptr = base_ptr;
            sum_r += *ptr * filter[1][1]; ptr++;
            sum_g += *ptr * filter[1][1]; ptr++;
            sum_b += *ptr * filter[1][1];
            
            ptr = base_ptr + input_jpeg.num_channels;
            sum_r += *ptr * filter[1][2]; ptr++;
            sum_g += *ptr * filter[1][2]; ptr++;
            sum_b += *ptr * filter[1][2];
            
            //lower row
            ptr = base_ptr + input_jpeg.width * input_jpeg.num_channels - input_jpeg.num_channels;
            sum_r += *ptr * filter[2][0]; ptr++;
            sum_g += *ptr * filter[2][0]; ptr++;
            sum_b += *ptr * filter[2][0];
            
            ptr = base_ptr + input_jpeg.width * input_jpeg.num_channels;
            sum_r += *ptr * filter[2][1]; ptr++;
            sum_g += *ptr * filter[2][1]; ptr++;
            sum_b += *ptr * filter[2][1];
            
            ptr = base_ptr + input_jpeg.width * input_jpeg.num_channels + input_jpeg.num_channels;
            sum_r += *ptr * filter[2][2]; ptr++;
            sum_g += *ptr * filter[2][2]; ptr++;
            sum_b += *ptr * filter[2][2];

            int idx = ((height - startRow) * input_jpeg.width + width) * input_jpeg.num_channels;
            filteredImage[idx] = static_cast<unsigned char>(std::round(sum_r));
            filteredImage[idx + 1] = static_cast<unsigned char>(std::round(sum_g));
            filteredImage[idx + 2] = static_cast<unsigned char>(std::round(sum_b));
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (taskid == MASTER) {
        unsigned char* finalImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
        
        // Copy data calculated by the master to the finalImage
        int size = input_jpeg.width * (endRow - startRow + 1) * input_jpeg.num_channels;
        std::copy(filteredImage, filteredImage + size, finalImage + startRow * input_jpeg.width * input_jpeg.num_channels);
        
        // Receive data calculated by the other processes
        for (int task = 1; task < numtasks; task++) {
            int start = task * rowsPerTask + std::min(task, remainingRows);
            int end = start + rowsPerTask - 1;
            if (task < remainingRows) end++;

            int recv_size = input_jpeg.width * (end - start + 1) * input_jpeg.num_channels;
            MPI_Recv(finalImage + start * input_jpeg.width * input_jpeg.num_channels, recv_size, MPI_UNSIGNED_CHAR, task, TAG_COMPUTE, MPI_COMM_WORLD, &status);
        }
        
        // Save the final image
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{finalImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG\n";
            delete[] finalImage;
            MPI_Finalize();
            return -1;
        }
        
        delete[] finalImage;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    } else {
        int send_size = input_jpeg.width * (endRow - startRow + 1) * input_jpeg.num_channels;
        MPI_Send(filteredImage, send_size, MPI_UNSIGNED_CHAR, MASTER, TAG_COMPUTE, MPI_COMM_WORLD);
    }

    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    
    MPI_Finalize();
    return 0;
}
