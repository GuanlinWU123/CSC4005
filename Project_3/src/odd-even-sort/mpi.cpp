//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Odd-Even Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0

void oddEvenSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    /* Your code here!
       Implement parallel odd-even sort with MPI
    */
    
    int n = vec.size();
    int local_n = n / numtasks;
    int remainder = n % numtasks;
    int start = taskid * local_n + std::min(taskid, remainder);
    int end = start + local_n + (taskid < remainder ? 1 : 0);
    std::vector<int> local_vec(vec.begin() + start, vec.begin() + end);

    bool global_sorted = false;
    while (!global_sorted) {
        bool local_sorted = true;

        // Perform local odd-even sort
        for (int phase = 0; phase <= 1; ++phase) {
            for (int i = phase; i < local_vec.size() - 1; i += 2) {
                if (local_vec[i] > local_vec[i + 1]) {
                    std::swap(local_vec[i], local_vec[i + 1]);
                    local_sorted = false;
                }
            }
        }

        // Exchange data with neighboring processes
        int left_neighbor = taskid - 1;
        int right_neighbor = taskid + 1;
        int left_data, right_data;
        if (left_neighbor >= 0) {
            MPI_Sendrecv(&local_vec[0], 1, MPI_INT, left_neighbor, 0, &left_data, 1, MPI_INT, left_neighbor, 0, MPI_COMM_WORLD, status);
            if (left_data > local_vec[0]) {
                local_vec[0] = left_data;
                local_sorted = false;
            }
        }
        if (right_neighbor < numtasks) {
            MPI_Sendrecv(&local_vec.back(), 1, MPI_INT, right_neighbor, 0, &right_data, 1, MPI_INT, right_neighbor, 0, MPI_COMM_WORLD, status);
            if (right_data < local_vec.back()) {
                local_vec.back() = right_data;
                local_sorted = false;
            }
        }

        // Check if the entire vector is sorted
        MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }

    // Gather sorted sublists from each process
    if (taskid < remainder) {
        MPI_Gather(&local_vec[0], local_n + 1, MPI_INT, &vec[0], local_n + 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&local_vec[0], local_n, MPI_INT, &vec[0], local_n, MPI_INT, MASTER, MPI_COMM_WORLD);
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

    oddEvenSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Odd-Even Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}