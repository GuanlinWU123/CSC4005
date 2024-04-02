#include "simple_ml_openacc.hpp"

void matrix_dot_openacc(const float *A, const float *B, float *C, size_t m,
                        size_t n, size_t k) {
  // BEGIN YOUR CODE
  memset(C, 0, sizeof(float) * m * k);

#pragma acc data copyin(A[0 : m * n], B[0 : n * k]) copyout(C[0 : m * k])
{
  #pragma acc parallel loop collapse(2)
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < k; ++j) {
      float sum = 0.0;
      #pragma acc loop seq
      for (size_t p = 0; p < n; ++p) {
        sum += A[i * n + p] * B[p * k + j];
      }
      C[i * k + j] = sum;
    }
  }
}

  // END YOUR CODE
}

void matrix_dot_trans_openacc(const float *A, const float *B, float *C,
                              size_t n, size_t m, size_t k) {
  // BEGIN YOUR CODE
  memset(C, 0, sizeof(float) * m * k);

#pragma acc data copyin(A[0 : m * n], B[0 : n * k]) copyout(C[0 : m * k])
{
  #pragma acc parallel loop collapse(2)
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < k; ++j) {
      float sum = 0;
      #pragma acc loop seq
      for (size_t p = 0; p < n; ++p) {
        sum += A[p * m + i] * B[p * k + j];
      }
      C[i * k + j] = sum;
    }
  }
}
  // END YOUR CODE
}

void matrix_trans_dot_openacc(const float *A, const float *B, float *C,
                              size_t m, size_t n, size_t k) {
  // BEGIN YOUR CODE
  memset(C, 0, sizeof(float) * m * k);

  // #pragma omp parallel for
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < k; ++j) {
      for (size_t p = 0; p < n; ++p) {
        C[i * k + j] += A[i * n + p] * B[j * n + p];
      }
    }
  }

  // END YOUR CODE
}

void matrix_minus_openacc(float *A, const float *B, size_t m, size_t n) {
  // BEGIN YOUR CODE
  // #pragma omp target parallel for map(from: A[0:m * n]) map(to: B[0:m*n])
  for (size_t i = 0; i < m * n; ++i) {
    A[i] -= B[i];
  }

  // END YOUR CODE
}

void matrix_mul_scalar_openacc(float *C, float scalar, size_t m, size_t n) {
  // BEGIN YOUR CODE
  // #pragma omp target parallel for map(from: C[0:m * n]) map(to: scalar)
  for (size_t i = 0; i < m * n; ++i) {
    C[i] *= scalar;
  }
  // END YOUR CODE
}

void matrix_div_scalar_openacc(float *C, float scalar, size_t m, size_t n) {
  // BEGIN YOUR CODE
  // #pragma omp target parallel for map(from: C[0:m * n]) map(to: scalar)
  for (size_t i = 0; i < m * n; ++i) {
    C[i] /= scalar;
  }
  // END YOUR CODE
}

void matrix_softmax_normalize_openacc(float *C, size_t m, size_t n) {
  // BEGIN YOUR CODE
  for (size_t i = 0; i < m; ++i) {
    float sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      C[i * n + j] = exp(C[i * n + j]); // Exponentiate each logit
      sum += C[i * n + j];
    }
    for (size_t j = 0; j < n; ++j) {
      C[i * n + j] /= sum; // Normalize
    }
  }
  // END YOUR CODE
}

void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y,
                                      size_t m, size_t k) {
  // BEGIN YOUR CODE
  std::fill(Y, Y + m * k, 0.0f);

  // Set the appropriate element for each row to 1
  // #pragma omp target parallel for map(from: Y[0:m * k]) map(to: y[0:m])
  for (size_t i = 0; i < m; ++i) {
    size_t label = y[i];
    if (label < k) {
      Y[i * k + label] = 1.0f;
    }
  }
  // END YOUR CODE
}

void softmax_regression_epoch_openacc(const float *X, const unsigned char *y,
                                      float *theta, size_t m, size_t n,
                                      size_t k, float lr, size_t batch) {
  // BEGIN YOUR CODE
  for (size_t i = 0; i < m; i += batch) {
    size_t current_batch_size = std::min(batch, m - i);
    float *logits = new float[current_batch_size * k];
    float *Y = new float[current_batch_size * k];
    float *grad = new float[n * k];
    memset(grad, 0, n * k * sizeof(float));

    // Convert the label vector y to a one-hot encoded matrix Y for the
    // current batch
    vector_to_one_hot_matrix_openacc(y + i, Y, current_batch_size, k);

    // Forward pass: Compute logits for the current batch
    matrix_dot_openacc(X + i * n, theta, logits, current_batch_size, n, k);

    // Apply softmax normalization on logits
    matrix_softmax_normalize_openacc(logits, current_batch_size, k);

    // Compute gradient for the current batch
    float *logits_minus_Y = new float[current_batch_size * k];
    memcpy(logits_minus_Y, logits, current_batch_size * k * sizeof(float));
    matrix_minus_openacc(logits_minus_Y, Y, current_batch_size, k);
    matrix_dot_trans_openacc(X + i * n, logits_minus_Y, grad,
                             current_batch_size, n, k);
    matrix_div_scalar_openacc(grad, static_cast<float>(current_batch_size), n,
                              k);

    // Update theta: theta -= lr * grad
    matrix_mul_scalar_openacc(grad, lr, n, k);
    matrix_minus_openacc(theta, grad, n, k);

    delete[] logits;
    delete[] Y;
    delete[] grad;
    delete[] logits_minus_Y;
    // END YOUR CODE
  }
}

void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data,
                           size_t num_classes, size_t epochs, float lr,
                           size_t batch) {
  /*
  Example function to fully train a softmax regression classifier
  */
  size_t size = train_data->input_dim * num_classes;
  float *theta = new float[size];
  memset(theta, 0, size * sizeof(float));
  size_t size_tr = train_data->images_num * num_classes;
  size_t size_te = test_data->images_num * num_classes;
  float *train_result = new float[size_tr];
  float *test_result = new float[size_te];
  float train_loss, train_err, test_loss, test_err;
  std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |"
            << std::endl;
  std::chrono::milliseconds elapsed_time;
  auto start_time = std::chrono::high_resolution_clock::now();
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    // BEGIN YOUR CODE
      softmax_regression_epoch_openacc(
          train_data->images_matrix,
          train_data->labels_array, theta, train_data->images_num,
          train_data->input_dim, num_classes, lr, batch);

    // After updating theta, compute predictions on train and test data
    matrix_dot_openacc(train_data->images_matrix, theta, train_result,
                       train_data->images_num, train_data->input_dim,
                       num_classes);
    matrix_dot_openacc(test_data->images_matrix, theta, test_result,
                       test_data->images_num, test_data->input_dim,
                       num_classes);

    // END YOUR CODE
    train_loss =
        mean_softmax_loss_openacc(train_result, train_data->labels_array,
                                  train_data->images_num, num_classes);
    test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array,
                                          test_data->images_num, num_classes);
    train_err = mean_err_openacc(train_result, train_data->labels_array,
                                 train_data->images_num, num_classes);
    test_err = mean_err_openacc(test_result, test_data->labels_array,
                                test_data->images_num, num_classes);
    std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
              << std::fixed << std::setprecision(5) << train_loss << " |   "
              << std::fixed << std::setprecision(5) << train_err << " |   "
              << std::fixed << std::setprecision(5) << test_loss << " |  "
              << std::fixed << std::setprecision(5) << test_err << " |"
              << std::endl;
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
  delete[] theta;
  delete[] train_result;
  delete[] test_result;
}

float mean_softmax_loss_openacc(const float *result,
                                const unsigned char *labels_array,
                                size_t images_num, size_t num_classes) {
  // BEGIN YOUR CODE
  float loss = 0.0;
  //   #pragma omp parallel for
  for (size_t i = 0; i < images_num; ++i) {
    float sum_exp = 0.0;
    for (size_t j = 0; j < num_classes; ++j) {
      sum_exp += exp(result[i * num_classes + j]);
    }
    loss += -result[i * num_classes + labels_array[i]] + log(sum_exp);
  }
  return loss / images_num;
  // END YOUR CODE
}

float mean_err_openacc(const float *result, const unsigned char *labels_array,
                       size_t images_num, size_t num_classes) {
  // BEGIN YOUR CODE
  size_t incorrect_count = 0;
  //   #pragma omp parallel for
  for (size_t i = 0; i < images_num; ++i) {
    size_t predicted_class = 0;
    for (size_t j = 1; j < num_classes; ++j) {
      if (result[i * num_classes + j] >
          result[i * num_classes + predicted_class]) {
        predicted_class = j;
      }
    }
    if (predicted_class != labels_array[i]) {
      ++incorrect_count;
    }
  }
  return static_cast<float>(incorrect_count) / images_num;
  // END YOUR CODE
}

void matrix_mul_openacc(float *A, const float *B, size_t size) {
  // BEGIN YOUR CODE
  //   #pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    A[i] *= B[i];
  }
  // END YOUR CODE
}

void nn_epoch_openacc(const float *X, const unsigned char *y, float *W1,
                      float *W2, size_t m, size_t n, size_t l, size_t k,
                      float lr, size_t batch) {
  // BEGIN YOUR CODE
  for (size_t i = 0; i < m; i += batch) {
    size_t current_batch_size = std::min(batch, m - i);

    float *Z1 = new float[current_batch_size * l];
    matrix_dot_openacc(X + i * n, W1, Z1, current_batch_size, n, l);
    for (size_t idx = 0; idx < current_batch_size * l; idx++) {
      if (Z1[idx] < 0) {
        Z1[idx] = 0.0;
      }
    }

    float *Z1_exp = new float[current_batch_size * k];
    matrix_dot_openacc(Z1, W2, Z1_exp, current_batch_size, l, k);

    for (size_t idx = 0; idx < current_batch_size * k; idx++) {
      Z1_exp[idx] = exp(Z1_exp[idx]);
    }

    float *Z2 = new float[current_batch_size * k];
    for (size_t row = 0; row < current_batch_size; row++) {
      float row_sum = 0;
      for (size_t col = 0; col < k; col++) {
        row_sum += Z1_exp[row * k + col];
      }
      for (size_t col = 0; col < k; col++) {
        Z2[row * k + col] = Z1_exp[row * k + col] / row_sum;
      }
    }

    float *Y = new float[current_batch_size * k];
    vector_to_one_hot_matrix_openacc(y + i, Y, current_batch_size, k);

    float *Z2_minus_Y = new float[current_batch_size * k];
    memcpy(Z2_minus_Y, Z2, current_batch_size * k * sizeof(float));
    matrix_minus_openacc(Z2_minus_Y, Y, current_batch_size, k);

    float *G1 = new float[current_batch_size * l];
    matrix_trans_dot_openacc(Z2_minus_Y, W2, G1, current_batch_size, k, l);
    for (size_t idx = 0; idx < current_batch_size * l; ++idx) {
      G1[idx] = (Z1[idx] > 0) ? G1[idx] : 0;
    }

    float *W1_1 = new float[n * l];
    matrix_dot_trans_openacc(X + i * n, G1, W1_1, current_batch_size, n, l);
    matrix_div_scalar_openacc(W1_1, current_batch_size, n, l);
    matrix_mul_scalar_openacc(W1_1, lr, n, l);

    float *W2_1 = new float[l * k];
    matrix_dot_trans_openacc(Z1, Z2_minus_Y, W2_1, current_batch_size, l, k);
    matrix_div_scalar_openacc(W2_1, current_batch_size, l, k);
    matrix_mul_scalar_openacc(W2_1, lr, l, k);

    matrix_minus_openacc(W1, W1_1, n, l);
    matrix_minus_openacc(W2, W2_1, l, k);
  }
  // END YOUR CODE
}

/**
 * Apply ReLU function on a matrix
 * Args:
 *     A (float*): Input matrix of size m * n
 *     B (float*): Output matrix of size m * n
 *     m (size_t): Number of rows in the matrices
 *     n (size_t): Number of columns in the matrices
 **/
static void apply_relu(const float *A, float *B, size_t m, size_t n) {
  size_t mn = m * n;
  for (size_t i = 0; i < mn; ++i) {
    B[i] = (A[i] > 0) ? A[i] : 0;
  }
}

void train_nn_openacc(const DataSet *train_data, const DataSet *test_data,
                      size_t num_classes, size_t hidden_dim, size_t epochs,
                      float lr, size_t batch) {
  size_t size_w1 = train_data->input_dim * hidden_dim;
  size_t size_w2 = hidden_dim * num_classes;
  float *W1 = new float[size_w1];
  float *W2 = new float[size_w2];
  std::mt19937 rng;
  rng.seed(0);
  std::normal_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < size_w1; i++) {
    W1[i] = dist(rng);
  }
  for (size_t i = 0; i < size_w2; i++) {
    W2[i] = dist(rng);
  }
  matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
  matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);
  size_t size_tr = train_data->images_num * num_classes;
  size_t size_te = test_data->images_num * num_classes;
  float *train_result = new float[size_tr];
  float *test_result = new float[size_te];
  float train_loss, train_err, test_loss, test_err;
  std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |"
            << std::endl;
  std::chrono::milliseconds elapsed_time;
  auto start_time = std::chrono::high_resolution_clock::now();
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    // BEGIN YOUR CODE
    nn_epoch_openacc(train_data->images_matrix, train_data->labels_array, W1, W2,
                 train_data->images_num, train_data->input_dim, hidden_dim,
                 num_classes, lr, batch);
    // }

    float *G1 = new float[train_data->images_num * hidden_dim];
    matrix_dot_openacc(train_data->images_matrix, W1, G1, train_data->images_num,
               train_data->input_dim, hidden_dim);
    apply_relu(G1, G1, train_data->images_num, hidden_dim);
    matrix_dot_openacc(G1, W2, train_result, train_data->images_num, hidden_dim,
               num_classes);

    float *G2 = new float[test_data->images_num * hidden_dim];
    matrix_dot_openacc(test_data->images_matrix, W1, G2, test_data->images_num,
               test_data->input_dim, hidden_dim);
    apply_relu(G2, G2, test_data->images_num, hidden_dim);
    matrix_dot_openacc(G2, W2, test_result, test_data->images_num, hidden_dim,
               num_classes);
    // END YOUR CODE
    train_loss =
        mean_softmax_loss_openacc(train_result, train_data->labels_array,
                                  train_data->images_num, num_classes);
    test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array,
                                          test_data->images_num, num_classes);
    train_err = mean_err_openacc(train_result, train_data->labels_array,
                                 train_data->images_num, num_classes);
    test_err = mean_err_openacc(test_result, test_data->labels_array,
                                test_data->images_num, num_classes);
    std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
              << std::fixed << std::setprecision(5) << train_loss << " |   "
              << std::fixed << std::setprecision(5) << train_err << " |   "
              << std::fixed << std::setprecision(5) << test_loss << " |  "
              << std::fixed << std::setprecision(5) << test_err << " |"
              << std::endl;
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
  delete[] W1;
  delete[] W2;
  delete[] train_result;
  delete[] test_result;
}
