#include <iostream>
#include <cmath>
#include <chrono>

#include "utils.hpp"
#include <immintrin.h>  // Include for SIMD intrinsics

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

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

    // Create the output image buffer
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;

    // Prepross, store reds, greens and blues separately
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto& num_channels = input_jpeg.num_channels;
    auto& buffer = input_jpeg.buffer;
    auto& width = input_jpeg.width;
    auto& height = input_jpeg.height;

    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Mask used for shuffling when store int32s to u_int8 arrays
    __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Optimize the nested for loop with SIMD
    for (int h = 1; h < height - 1; h++)
    {
        for (int w = 1; w < width - 1; w++)
        {
            __m256 sum_r = _mm256_setzero_ps();
            __m256 sum_g = _mm256_setzero_ps();
            __m256 sum_b = _mm256_setzero_ps();

            int index = (h * width + w) * num_channels;
            int up_index = index - width * num_channels;
            int down_index = index + width * num_channels;
            
            // upper row
            __m256 filter_val_1 = _mm256_set1_ps(filter[0][0]);
            int channel_value_r_1 = buffer[up_index - num_channels];
            int channel_value_g_1 = buffer[up_index - num_channels + 1];
            int channel_value_b_1 = buffer[up_index - num_channels + 2];
            __m256 channel_val_r_1 = _mm256_set1_ps(channel_value_r_1);
            __m256 channel_val_g_1 = _mm256_set1_ps(channel_value_g_1);
            __m256 channel_val_b_1 = _mm256_set1_ps(channel_value_b_1);
            sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(channel_val_r_1, filter_val_1));
            sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(channel_val_g_1, filter_val_1));
            sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(channel_val_b_1, filter_val_1));

            __m256 filter_val_2 = _mm256_set1_ps(filter[0][1]);
            int channel_value_r_2 = buffer[up_index];
            int channel_value_g_2 = buffer[up_index + 1];
            int channel_value_b_2 = buffer[up_index + 2];
            __m256 channel_val_r_2 = _mm256_set1_ps(channel_value_r_2);
            __m256 channel_val_g_2 = _mm256_set1_ps(channel_value_g_2);
            __m256 channel_val_b_2 = _mm256_set1_ps(channel_value_b_2);
            sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(channel_val_r_2, filter_val_2));
            sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(channel_val_g_2, filter_val_2));
            sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(channel_val_b_2, filter_val_2));

            __m256 filter_val_3 = _mm256_set1_ps(filter[0][2]);
            int channel_value_r_3 = buffer[up_index + num_channels];
            int channel_value_g_3 = buffer[up_index + num_channels + 1];
            int channel_value_b_3 = buffer[up_index + num_channels + 2];
            __m256 channel_val_r_3 = _mm256_set1_ps(channel_value_r_3);
            __m256 channel_val_g_3 = _mm256_set1_ps(channel_value_g_3);
            __m256 channel_val_b_3 = _mm256_set1_ps(channel_value_b_3);
            sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(channel_val_r_3, filter_val_3));
            sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(channel_val_g_3, filter_val_3));
            sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(channel_val_b_3, filter_val_3));
            
            // middle row
            __m256 filter_val_4 = _mm256_set1_ps(filter[1][0]);
            int channel_value_r_4 = buffer[index - num_channels];
            int channel_value_g_4 = buffer[index - num_channels + 1];
            int channel_value_b_4 = buffer[index - num_channels + 2];
            __m256 channel_val_r_4 = _mm256_set1_ps(channel_value_r_4);
            __m256 channel_val_g_4 = _mm256_set1_ps(channel_value_g_4);
            __m256 channel_val_b_4 = _mm256_set1_ps(channel_value_b_4);
            sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(channel_val_r_4, filter_val_4));
            sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(channel_val_g_4, filter_val_4));
            sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(channel_val_b_4, filter_val_4));

            __m256 filter_val_5 = _mm256_set1_ps(filter[1][1]);
            int channel_value_r_5 = buffer[index];
            int channel_value_g_5 = buffer[index + 1];
            int channel_value_b_5 = buffer[index + 2];
            __m256 channel_val_r_5 = _mm256_set1_ps(channel_value_r_5);
            __m256 channel_val_g_5 = _mm256_set1_ps(channel_value_g_5);
            __m256 channel_val_b_5 = _mm256_set1_ps(channel_value_b_5);
            sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(channel_val_r_5, filter_val_5));
            sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(channel_val_g_5, filter_val_5));
            sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(channel_val_b_5, filter_val_5));

            __m256 filter_val_6 = _mm256_set1_ps(filter[1][2]);
            int channel_value_r_6 = buffer[index + num_channels];
            int channel_value_g_6 = buffer[index + num_channels + 1];
            int channel_value_b_6 = buffer[index + num_channels + 2];
            __m256 channel_val_r_6 = _mm256_set1_ps(channel_value_r_6);
            __m256 channel_val_g_6 = _mm256_set1_ps(channel_value_g_6);
            __m256 channel_val_b_6 = _mm256_set1_ps(channel_value_b_6);
            sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(channel_val_r_6, filter_val_6));
            sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(channel_val_g_6, filter_val_6));
            sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(channel_val_b_6, filter_val_6));

            // lower row
            __m256 filter_val_7 = _mm256_set1_ps(filter[2][0]);
            int channel_value_r_7 = buffer[down_index - num_channels];
            int channel_value_g_7 = buffer[down_index - num_channels + 1];
            int channel_value_b_7 = buffer[down_index - num_channels + 2];
            __m256 channel_val_r_7 = _mm256_set1_ps(channel_value_r_7);
            __m256 channel_val_g_7 = _mm256_set1_ps(channel_value_g_7);
            __m256 channel_val_b_7 = _mm256_set1_ps(channel_value_b_7);
            sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(channel_val_r_7, filter_val_7));
            sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(channel_val_g_7, filter_val_7));
            sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(channel_val_b_7, filter_val_7));

            __m256 filter_val_8 = _mm256_set1_ps(filter[2][1]);
            int channel_value_r_8 = buffer[down_index];
            int channel_value_g_8 = buffer[down_index + 1];
            int channel_value_b_8 = buffer[down_index + 2];
            __m256 channel_val_r_8 = _mm256_set1_ps(channel_value_r_8);
            __m256 channel_val_g_8 = _mm256_set1_ps(channel_value_g_8);
            __m256 channel_val_b_8 = _mm256_set1_ps(channel_value_b_8);
            sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(channel_val_r_8, filter_val_8));
            sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(channel_val_g_8, filter_val_8));
            sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(channel_val_b_8, filter_val_8));

            __m256 filter_val_9 = _mm256_set1_ps(filter[2][2]);
            int channel_value_r_9 = buffer[down_index + num_channels];
            int channel_value_g_9 = buffer[down_index + num_channels + 1];
            int channel_value_b_9 = buffer[down_index + num_channels + 2];
            __m256 channel_val_r_9 = _mm256_set1_ps(channel_value_r_9);
            __m256 channel_val_g_9 = _mm256_set1_ps(channel_value_g_9);
            __m256 channel_val_b_9 = _mm256_set1_ps(channel_value_b_9);
            sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(channel_val_r_9, filter_val_9));
            sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(channel_val_g_9, filter_val_9));
            sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(channel_val_b_9, filter_val_9));


            // for (int i = -1; i <= 1; i++) {
            //     for (int j = -1; j <= 1; j++) {
            //         int channel_value_r = input_jpeg.buffer[((height + i) * input_jpeg.width + (width + j)) * input_jpeg.num_channels];
            //         int channel_value_g = input_jpeg.buffer[((height + i) * input_jpeg.width + (width + j)) * input_jpeg.num_channels + 1];
            //         int channel_value_b = input_jpeg.buffer[((height + i) * input_jpeg.width + (width + j)) * input_jpeg.num_channels + 2];

            //         __m256 filter_val = _mm256_set1_ps(filter[i + 1][j + 1]);
            //         __m256 channel_val_r = _mm256_set1_ps(channel_value_r);
            //         __m256 channel_val_g = _mm256_set1_ps(channel_value_g);
            //         __m256 channel_val_b = _mm256_set1_ps(channel_value_b);
            //         // int *data = (int *)&channel_val_r;
            //         // for(int i = 0; i<8; i++){
            //         //     std::cout << data[i] << " ";
            //         // }
            //         // std::cout << std::endl;

            //         sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(channel_val_r, filter_val));
            //         sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(channel_val_g, filter_val));
            //         sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(channel_val_b, filter_val));
            //         // int *data = (int *)&sum_r;
            //         // for(int i = 0; i<8; i++){
            //         //     std::cout << data[i] << " ";
            //         // }
            //         // std::cout << std::endl;
            //     }
            // }

        float result_r[8], result_g[8], result_b[8];
        _mm256_storeu_ps(result_r, sum_r);
        _mm256_storeu_ps(result_g, sum_g);
        _mm256_storeu_ps(result_b, sum_b);

        // Accumulate the results
        float final_sum_r = result_r[0];
        float final_sum_g = result_g[0];
        float final_sum_b = result_b[0];
        // std::cout << final_sum_r << " " << final_sum_g << " " << final_sum_b << std::endl;

        // Write to the output
        filteredImage[(h * input_jpeg.width + w) * input_jpeg.num_channels] = std::max(0, std::min(225, static_cast<int>(final_sum_r)));
        filteredImage[(h * input_jpeg.width + w) * input_jpeg.num_channels + 1] = std::max(0, std::min(225, static_cast<int>(final_sum_g)));
        filteredImage[(h * input_jpeg.width + w) * input_jpeg.num_channels + 2] = std::max(0, std::min(225, static_cast<int>(final_sum_b)));
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}