#include <iostream>
#include <cmath>
#include <chrono>

#include "utils.hpp"
#include <openacc.h> // Include OpenACC header

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

// #pragma acc declare const(filter)

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);
    
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    
    // Apply the filter to the image
#pragma acc enter data copyin(filter)
#pragma acc update device(filter)

    // auto filteredImage = new unsigned char[width * height * num_channels];
    unsigned char *filteredImage = new unsigned char[width * height * num_channels];
    unsigned char *buffer = new unsigned char[width * height * num_channels];
    for (int i = 0; i < width * height * num_channels; i++)
    {
        buffer[i] = input_jpeg.buffer[i];
    }
    
// #pragma acc enter data copyin(input_jpeg.buffer[0 : width * height * num_channels], \
//                                  filteredImage[0 : width * height * num_channels])

// #pragma acc update device(input_jpeg.buffer[0 : width * height * num_channels], \
//                                  filteredImage[0 : width * height * num_channels])

#pragma acc enter data copyin(filteredImage[0 : width * height * num_channels], \
                                 buffer[0 : width * height * num_channels])

#pragma acc enter data copyin(filteredImage[0 : width * height * num_channels], \
                                 buffer[0 : width * height * num_channels])
    
    auto start_time = std::chrono::high_resolution_clock::now();
    // std::cout << "111111111111" << "\n";
    // Nested for loop, optimized with OpenACC
#pragma acc parallel loop present(filteredImage[0 : width * height * num_channels], \
                                    buffer[0 : width * height * num_channels]) 
    // num_gangs(1024)
    // {
#pragma acc loop independent
    for (int h = 1; h < height - 1; h++)
    {
#pragma acc loop independent
        for (int w = 1; w < width - 1; w++)
        {
            // int sum_r = 0, sum_g = 0, sum_b = 0;
// #pragma acc loop seq
//             for (int i = -1; i <= 1; i++)
//             {
// #pragma acc loop seq
//                 for (int j = -1; j <= 1; j++)
//                 {
//                     int channel_value_r = buffer[((h + i) * width + (w + j)) * num_channels];
//                     int channel_value_g = buffer[((h + i) * width + (w + j)) * num_channels + 1];
//                     int channel_value_b = buffer[((h + i) * width + (w + j)) * num_channels + 2];
//                     sum_r += channel_value_r * filter[i + 1][j + 1];
//                     sum_g += channel_value_g * filter[i + 1][j + 1];
//                     sum_b += channel_value_b * filter[i + 1][j + 1];
//                 }
//             }
//             // printf("222222222222");
//             filteredImage[(h * width + w) * num_channels]
//                 = static_cast<unsigned char>(std::round(sum_r));
//             filteredImage[(h * width + w) * num_channels + 1]
//                 = static_cast<unsigned char>(std::round(sum_g));
//             filteredImage[(h * width + w) * num_channels + 2]
//                 = static_cast<unsigned char>(std::round(sum_b));

            int sum_r = 0, sum_g = 0, sum_b = 0;
            int index = (h * width + w) * num_channels;
            int up_index = index - width * num_channels;
            int down_index = index + width * num_channels;
            
            // upper row
            sum_r += buffer[up_index - num_channels] * filter[0][0];
            sum_g += buffer[up_index - num_channels + 1] * filter[0][0];
            sum_b += buffer[up_index - num_channels + 2] * filter[0][0];
            
            sum_r += buffer[up_index] * filter[0][1];
            sum_g += buffer[up_index + 1] * filter[0][1];
            sum_b += buffer[up_index + 2] * filter[0][1];
            
            sum_r += buffer[up_index + num_channels] * filter[0][2];
            sum_g += buffer[up_index + num_channels + 1] * filter[0][2];
            sum_b += buffer[up_index + num_channels + 2] * filter[0][2];
            
            // middle row
            sum_r += buffer[index - num_channels] * filter[1][0];
            sum_g += buffer[index - num_channels + 1] * filter[1][0];
            sum_b += buffer[index - num_channels + 2] * filter[1][0];
            
            sum_r += buffer[index] * filter[1][1];
            sum_g += buffer[index + 1] * filter[1][1];
            sum_b += buffer[index + 2] * filter[1][1];
            
            sum_r += buffer[index + num_channels] * filter[1][2];
            sum_g += buffer[index + num_channels + 1] * filter[1][2];
            sum_b += buffer[index + num_channels + 2] * filter[1][2];
            
            // lower row
            sum_r += buffer[down_index - num_channels] * filter[2][0];
            sum_g += buffer[down_index - num_channels + 1] * filter[2][0];
            sum_b += buffer[down_index - num_channels + 2] * filter[2][0];
            
            sum_r += buffer[down_index] * filter[2][1];
            sum_g += buffer[down_index + 1] * filter[2][1];
            sum_b += buffer[down_index + 2] * filter[2][1];
            
            sum_r += buffer[down_index + num_channels] * filter[2][2];
            sum_g += buffer[down_index + num_channels + 1] * filter[2][2];
            sum_b += buffer[down_index + num_channels + 2] * filter[2][2];
            
            // Assigning the result
            filteredImage[index] = static_cast<unsigned char>(std::round(sum_r));
            filteredImage[index + 1] = static_cast<unsigned char>(std::round(sum_g));
            filteredImage[index + 2] = static_cast<unsigned char>(std::round(sum_b));
        }
    }
    // }
    auto end_time = std::chrono::high_resolution_clock::now();
    // std::cout << "222222222222" << "\n";
    #pragma acc update self(buffer[0 : width * height * num_channels], \
                                 filteredImage[0 : width * height * num_channels])

    #pragma acc exit data copyout(filteredImage[0 : width * height * num_channels])

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, width, height, num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] buffer;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}




