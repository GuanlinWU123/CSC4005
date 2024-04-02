//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Quick Sort with MPI
//


#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"
#include <climits>

#define MASTER 0

int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }

    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

void localQuickSort(std::vector<int> &vec, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(vec, low, high);
        localQuickSort(vec, low, pivotIndex - 1);
        localQuickSort(vec, pivotIndex + 1, high);
    }
}

void quickSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    /* Your code here!
       Implement parallel quick sort with MPI
    */

    int block_size = vec.size() / numtasks;
    int remainder = vec.size() % numtasks;
    int start_index = taskid * block_size + std::min(taskid, remainder);
    int end_index = start_index + block_size + (taskid < remainder);

    // Extract local block and sort it
    std::vector<int> local_vec(vec.begin() + start_index, vec.begin() + end_index);
    localQuickSort(local_vec, 0, local_vec.size() - 1);

    // Binary tree reduction
    int step = 1;
    while (step < numtasks) {
        if (taskid % (2 * step) == 0) {
            if (taskid + step < numtasks) {
                int offset = (taskid + step) * block_size + std::min(taskid + step, remainder);
                int recv_size = (taskid + 2 * step <= numtasks) ? block_size * step : vec.size() - offset;
                std::vector<int> received_data(recv_size);
                MPI_Recv(&received_data[0], recv_size, MPI_INT, taskid + step, 0, MPI_COMM_WORLD, status);
                std::vector<int> merged_data(local_vec.size() + received_data.size());
                std::merge(local_vec.begin(), local_vec.end(), received_data.begin(), received_data.end(), merged_data.begin());
                local_vec = merged_data;
            }
        } else {
            int dest = taskid - step;
            MPI_Send(&local_vec[0], local_vec.size(), MPI_INT, dest, 0, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    if (taskid == MASTER) {
        vec = local_vec;
    }
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
            );
    }
    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    const int size = atoi(argv[1]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    quickSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Quick Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}