#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>    // OpenMP header
#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);

    // Check if the read operation failed
    if (input_jpeg.buffer == nullptr) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Allocate memory for the output image
    auto outputImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

    auto start_time = std::chrono::high_resolution_clock::now();

    // Parallelized convolution loop
    #pragma omp parallel for collapse(2)
    for (int height = 1; height < input_jpeg.height - 1; height++)
    {
        for (int width = 1; width < input_jpeg.width - 1; width++)
        {
            int sum_r = 0, sum_g = 0, sum_b = 0;
            unsigned char *base_ptr = &input_jpeg.buffer[(height * input_jpeg.width + width) * input_jpeg.num_channels];

            // Upper row
            unsigned char *ptr = base_ptr - input_jpeg.width * input_jpeg.num_channels - input_jpeg.num_channels;
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

            // Middle row
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

            // Lower row
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

            int output_index = (height * input_jpeg.width + width) * input_jpeg.num_channels;
            outputImage[output_index] = static_cast<unsigned char>(std::round(sum_r));
            outputImage[output_index + 1] = static_cast<unsigned char>(std::round(sum_g));
            outputImage[output_index + 2] = static_cast<unsigned char>(std::round(sum_b));
        }
    }


    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{outputImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Memory cleanup
    delete[] input_jpeg.buffer;
    delete[] outputImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
