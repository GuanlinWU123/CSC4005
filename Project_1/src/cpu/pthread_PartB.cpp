#include <iostream>
#include <cmath>
#include <chrono>
#include <pthread.h>
#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

// Structure to pass data to each thread
struct ThreadData {
    unsigned char* input_buffer;
    unsigned char* output_buffer;
    int width;
    int height;
    int num_channels;
    int start;
    int end;
};

// Function to apply the filter to a portion of the image
void* applyFilter(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    int width = data->width;
    int num_channels = data->num_channels;
    unsigned char* buffer = data->input_buffer;

    for (int h = data->start; h < data->end; ++h) {
        for (int w = 1; w < data->width - 1; ++w) {
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
            // for (int i = -1; i <= 1; ++i) {
            //     for (int j = -1; j <= 1; ++j) {
            //         int channel_value_r = data->input_buffer[((h + i) * data->width + (w + j)) * data->num_channels];
            //         int channel_value_g = data->input_buffer[((h + i) * data->width + (w + j)) * data->num_channels + 1];
            //         int channel_value_b = data->input_buffer[((h + i) * data->width + (w + j)) * data->num_channels + 2];
            //         sum_r += channel_value_r * filter[i + 1][j + 1];
            //         sum_g += channel_value_g * filter[i + 1][j + 1];
            //         sum_b += channel_value_b * filter[i + 1][j + 1];
            //     }
            // }


            data->output_buffer[(h * data->width + w) * data->num_channels]
                = static_cast<unsigned char>(std::round(sum_r));
            data->output_buffer[(h * data->width + w) * data->num_channels + 1]
                = static_cast<unsigned char>(std::round(sum_g));
            data->output_buffer[(h * data->width + w) * data->num_channels + 2]
                = static_cast<unsigned char>(std::round(sum_b));
        }
    }

    return nullptr;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);

    // Allocate memory for the filtered image
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;

    // User-specified thread count
    int num_threads = std::stoi(argv[3]);; 

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    auto start_time = std::chrono::high_resolution_clock::now();

    int chunk_size = input_jpeg.height / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].input_buffer = input_jpeg.buffer;
        thread_data[i].output_buffer = filteredImage;
        thread_data[i].width = input_jpeg.width;
        thread_data[i].height = input_jpeg.height;
        thread_data[i].num_channels = input_jpeg.num_channels;
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? input_jpeg.height : (i + 1) * chunk_size;

        pthread_create(&threads[i], nullptr, applyFilter, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

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
