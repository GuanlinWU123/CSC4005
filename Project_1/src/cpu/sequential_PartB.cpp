#include <iostream>
#include <cmath>
#include <chrono>
#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

const double filter_value = 1.0 / 9.0; // As the filter has a constant value of 1/9

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
    auto& width = input_jpeg.width;
    auto& height = input_jpeg.height;
    auto& num_channels = input_jpeg.num_channels;
    auto& buffer = input_jpeg.buffer;
    
    // Apply the filter to the image
    auto filteredImage = new unsigned char[width * height * num_channels](); // Initialize with zero
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Unrolled Nested Loop
    for (int h = 1; h < height - 1; h++)
    {
        for (int w = 1; w < width - 1; w++)
        {
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

    // for (int h = 1; h < height - 1; h++) {
    //     for (int w = 1; w < width - 1; w++) {
    //         int sum_r = 0, sum_g = 0, sum_b = 0;
    //         int index = (h * width + w) * num_channels;
    //         int up_index = index - width * num_channels;
    //         int down_index = index + width * num_channels;

    //         // Define temporary variables
    //         unsigned char *up_ptr = &buffer[up_index];
    //         unsigned char *mid_ptr = &buffer[index];
    //         unsigned char *down_ptr = &buffer[down_index];
    //         unsigned char *target_ptr = &filteredImage[index];

    //         // Unroll upper row calculations
    //         sum_r += up_ptr[-num_channels] * filter[0][0];
    //         sum_g += up_ptr[-num_channels + 1] * filter[0][0];
    //         sum_b += up_ptr[-num_channels + 2] * filter[0][0];
    //         sum_r += up_ptr[0] * filter[0][1];
    //         sum_g += up_ptr[1] * filter[0][1];
    //         sum_b += up_ptr[2] * filter[0][1];
    //         sum_r += up_ptr[num_channels] * filter[0][2];
    //         sum_g += up_ptr[num_channels + 1] * filter[0][2];
    //         sum_b += up_ptr[num_channels + 2] * filter[0][2];

    //         // Unroll middle row calculations
    //         sum_r += mid_ptr[-num_channels] * filter[1][0];
    //         sum_g += mid_ptr[-num_channels + 1] * filter[1][0];
    //         sum_b += mid_ptr[-num_channels + 2] * filter[1][0];
    //         sum_r += mid_ptr[0] * filter[1][1];
    //         sum_g += mid_ptr[1] * filter[1][1];
    //         sum_b += mid_ptr[2] * filter[1][1];
    //         sum_r += mid_ptr[num_channels] * filter[1][2];
    //         sum_g += mid_ptr[num_channels + 1] * filter[1][2];
    //         sum_b += mid_ptr[num_channels + 2] * filter[1][2];

    //         // Unroll lower row calculations
    //         sum_r += down_ptr[-num_channels] * filter[2][0];
    //         sum_g += down_ptr[-num_channels + 1] * filter[2][0];
    //         sum_b += down_ptr[-num_channels + 2] * filter[2][0];
    //         sum_r += down_ptr[0] * filter[2][1];
    //         sum_g += down_ptr[1] * filter[2][1];
    //         sum_b += down_ptr[2] * filter[2][1];
    //         sum_r += down_ptr[num_channels] * filter[2][2];
    //         sum_g += down_ptr[num_channels + 1] * filter[2][2];
    //         sum_b += down_ptr[num_channels + 2] * filter[2][2];

    //         // Assigning the result
    //         target_ptr[0] = static_cast<unsigned char>(std::round(sum_r));
    //         target_ptr[1] = static_cast<unsigned char>(std::round(sum_g));
    //         target_ptr[2] = static_cast<unsigned char>(std::round(sum_b));
    //     }
    // }

    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, width, height, num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    
    // Post-processing
    delete[] buffer;
    delete[] filteredImage;
    
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
