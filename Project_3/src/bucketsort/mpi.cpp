//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0

void insertionSort(std::vector<int>& bucket) {
    for (int i = 1; i < bucket.size(); ++i) {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key) {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}

void bucketSort(std::vector<int>& vec, int num_buckets, int numtasks, int taskid, MPI_Status* status) {

    int vec_size = vec.size();

    if (!vec_size)
    {
        return;
    }

    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());

    int range = max_val - min_val + 1;
    int bucket_range = range / num_buckets + 1;
    int max_bucket_size = vec_size / num_buckets + vec_size % num_buckets;

    std::vector<int> buckets_per_process(numtasks);
    std::vector<int> size_sdispls(numtasks, 0);
    int unit_buckets = num_buckets / numtasks;
    for (int i = 0; i < numtasks; i++)
    {
        buckets_per_process[i] = unit_buckets;
        if (i == numtasks - 1)
        {
            buckets_per_process[i] += num_buckets % numtasks;
        }
        if (i > 0)
        {
            size_sdispls[i] = size_sdispls[i - 1] + buckets_per_process[i - 1];
        }
    }

    // 1. Data Distribution
    int chunk_size = vec_size / numtasks;
    auto begin_iter = vec.begin() + chunk_size * taskid;
    auto end_iter =
        (taskid == numtasks - 1) ? vec.end() : begin_iter + chunk_size;
    int process_size = end_iter - begin_iter;

    // avoid scatter to speed up
    // 2. Local Bucketing
    std::vector<std::vector<int>> local_buckets(num_buckets);
    int estimate_size = process_size / num_buckets * 5;
    // avoid eager allocation to speed up
    for (auto& p : local_buckets)
    {
        p.reserve(estimate_size);
    }
    for (auto num = begin_iter; num < end_iter; num++)
    {
        int bucket_idx = (*num - min_val) / bucket_range;
        bucket_idx = bucket_idx - bucket_idx / num_buckets;
        local_buckets[bucket_idx].push_back(*num);
    }

    // 3. Bucket Distribution
    std::vector<int> bucket_sizes(num_buckets);
    for (int i = 0; i < local_buckets.size(); i++)
    {
        auto& bucket = local_buckets[i];
        bucket_sizes[i] = bucket.size();
    }

    // no place here

    int current_bucket_num = buckets_per_process[taskid];
    std::vector<int> alien_bucket_sizes(numtasks * current_bucket_num);
    std::vector<int> size_recvcounts(numtasks, current_bucket_num);
    std::vector<int> size_rdispls(numtasks, 0);

    for (int i = 1; i < numtasks; i++)
    {
        size_rdispls[i] = i * current_bucket_num;
    }

    MPI_Alltoallv(&bucket_sizes[0], &buckets_per_process[0], &size_sdispls[0],
                  MPI_INT, &alien_bucket_sizes[0], &size_recvcounts[0],
                  &size_rdispls[0], MPI_INT, MPI_COMM_WORLD);

    std::vector<int> data_offset(numtasks * current_bucket_num + 1, 0);
    std::partial_sum(alien_bucket_sizes.begin(), alien_bucket_sizes.end(),
                     data_offset.begin() + 1);

    std::vector<int> send_data;
    for (auto& vec : local_buckets)
    {
        for (auto& num : vec)
        {
            send_data.push_back(num);
        }
    }

    std::vector<int> data_sendcouts(numtasks, 0);
    std::vector<int> data_displs(numtasks, 0);
    for (int i = 0; i < numtasks; i++)
    {
        for (int j = 0; j < buckets_per_process[i]; j++)
        {
            data_sendcouts[i] += bucket_sizes[i * unit_buckets + j];
        }
    }

    for (int i = 1; i < numtasks; i++)
    {
        data_displs[i] = data_displs[i - 1] + data_sendcouts[i - 1];
    }

    std::vector<int> data_recvcounts(numtasks, 0);
    std::vector<int> data_rdispls(numtasks, 0);
    for (int i = 1; i < numtasks; i++)
    {
        data_rdispls[i] = data_offset[i * current_bucket_num];
    }
    for (int i = 0; i < numtasks; i++)
    {
        data_recvcounts[i] =
            data_offset[(i + 1) * current_bucket_num] - data_rdispls[i];
    }

    std::vector<int> recv_data(data_offset.back());

    MPI_Alltoallv(&send_data[0], &data_sendcouts[0], &data_displs[0], MPI_INT,
                  &recv_data[0], &data_recvcounts[0], &data_rdispls[0], MPI_INT,
                  MPI_COMM_WORLD);

    int bucket_id_start = taskid * unit_buckets;

    for (int i = 0; i < current_bucket_num; i++)
    {
        for (int j = 0; j < numtasks; j++)
        {
            if (j == taskid) continue;
            local_buckets[i + bucket_id_start].insert(
                local_buckets[i + bucket_id_start].end(),
                recv_data.begin() + data_offset[j * current_bucket_num + i],
                recv_data.begin() +
                    data_offset[j * current_bucket_num + i + 1]);
        }
    }

    // 4. Local Sorting
    int concerned_size = 0;
    std::vector<int> concerned_buckets;
    for (int i = bucket_id_start; i < bucket_id_start + current_bucket_num; i++)
    {
        insertionSort(local_buckets[i]);
        concerned_buckets.insert(concerned_buckets.end(),
                                 local_buckets[i].begin(),
                                 local_buckets[i].end());
    }
    concerned_size = concerned_buckets.size();

    // 5. Global Bucketing
    std::vector<int> result_sizes(numtasks);
    MPI_Gather(&concerned_size, 1, MPI_INT, &result_sizes[0], 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    std::vector<int> final_displs(numtasks + 1, 0);
    std::partial_sum(result_sizes.begin(), result_sizes.end(),
                     final_displs.begin() + 1);

    MPI_Gatherv(&concerned_buckets[0], concerned_size, MPI_INT, &vec[0],
                &result_sizes[0], &final_displs[0], MPI_INT, 0, MPI_COMM_WORLD);

}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size bucket_num\n"
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

    const int bucket_num = atoi(argv[2]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    bucketSort(vec, bucket_num, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}