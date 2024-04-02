#include "simple_ml_ext.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <math.h>

DataSet::DataSet(size_t images_num, size_t input_dim)
    : images_num(images_num), input_dim(input_dim) {
  images_matrix = new float[images_num * input_dim];
  labels_array = new unsigned char[images_num];
}

DataSet::~DataSet() {
  delete[] images_matrix;
  delete[] labels_array;
}

uint32_t swap_endian(uint32_t val) {
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | (val >> 16);
}

/**
 *Read an images and labels file in MNIST format.  See this page:
 *http://yann.lecun.com/exdb/mnist/ for a description of the file format.
 *Args:
 *    image_filename (str): name of images file in MNIST format (idx3-ubyte)
 *    label_filename (str): name of labels file in MNIST format (idx1-ubyte)
 **/
DataSet *parse_mnist(const std::string &image_filename,
                     const std::string &label_filename) {
  std::ifstream images_file(image_filename, std::ios::in | std::ios::binary);
  std::ifstream labels_file(label_filename, std::ios::in | std::ios::binary);
  uint32_t magic_num, images_num, rows_num, cols_num;

  images_file.read(reinterpret_cast<char *>(&magic_num), 4);
  labels_file.read(reinterpret_cast<char *>(&magic_num), 4);

  images_file.read(reinterpret_cast<char *>(&images_num), 4);
  labels_file.read(reinterpret_cast<char *>(&images_num), 4);
  images_num = swap_endian(images_num);

  images_file.read(reinterpret_cast<char *>(&rows_num), 4);
  rows_num = swap_endian(rows_num);
  images_file.read(reinterpret_cast<char *>(&cols_num), 4);
  cols_num = swap_endian(cols_num);

  DataSet *dataset = new DataSet(images_num, rows_num * cols_num);

  labels_file.read(reinterpret_cast<char *>(dataset->labels_array), images_num);
  unsigned char *pixels = new unsigned char[images_num * rows_num * cols_num];
  images_file.read(reinterpret_cast<char *>(pixels),
                   images_num * rows_num * cols_num);
  for (size_t i = 0; i < images_num * rows_num * cols_num; i++) {
    dataset->images_matrix[i] = static_cast<float>(pixels[i]) / 255;
  }

  delete[] pixels;

  return dataset;
}

/**
 *Print Matrix
 *Print the elements of a matrix A with size m * n.
 *Args:
 *      A (float*): Matrix of size m * n
 **/
void print_matrix(float *A, size_t m, size_t n) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      std::cout << A[i * n + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

/**
 * Matrix Dot Multiplication
 * Efficiently compute C = A.dot(B)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot(const float *A, const float *B, float *C, size_t m, size_t n,
                size_t k) {
  // BEGIN YOUR CODE
  memset(C, 0, sizeof(float) * m * k);
  for (size_t i = 0; i < m; ++i) {
    for (size_t p = 0; p < n; ++p) {
      for (size_t j = 0; j < k; ++j) {
        C[i * k + j] += A[i * n + p] * B[p * k + j];
      }
    }
  }
  // END YOUR CODE
}

/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size n * m
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot_trans(const float *A, const float *B, float *C, size_t n,
                      size_t m, size_t k) {
  // BEGIN YOUR CODE
  memset(C, 0, sizeof(float) * m * k);

  for (size_t p = 0; p < n; ++p) {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < k; ++j) {
        C[i * k + j] += A[p * m + i] * B[p * k + j];
      }
    }
  }
  // END YOUR CODE
}

/**
 * Matrix Dot Multiplication Trans Version 2
 * Efficiently compute C = A.dot(B.T)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size k * n
 *     C (float*): Matrix of size m * k
 **/
void matrix_trans_dot(const float *A, const float *B, float *C, size_t m,
                      size_t n, size_t k) {
  // BEGIN YOUR CODE
  memset(C, 0, sizeof(float) * m * k);

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < k; ++j) {
      for (size_t p = 0; p < n; ++p) {
        C[i * k + j] += A[i * n + p] * B[j * n + p];
      }
    }
  }
  // END YOUR CODE
}

/**
 * Matrix Minus
 * Efficiently compute A = A - B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_minus(float *A, const float *B, size_t m, size_t n) {
  // BEGIN YOUR CODE
  for (size_t i = 0; i < m * n; ++i) {
    A[i] -= B[i];
  }
  // END YOUR CODE
}

/**
 * Matrix Multiplication Scalar
 * For each element C[i] of C, C[i] *= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_mul_scalar(float *C, float scalar, size_t m, size_t n) {
  // BEGIN YOUR CODE
  for (size_t i = 0; i < m * n; ++i) {
    C[i] *= scalar;
  }
  // END YOUR CODE
}

/**
 * Matrix Division Scalar
 * For each element C[i] of C, C[i] /= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_div_scalar(float *C, float scalar, size_t m, size_t n) {
  // BEGIN YOUR CODE
  for (size_t i = 0; i < m * n; ++i) {
    C[i] /= scalar;
  }
  // END YOUR CODE
}

/**
 * Matrix Softmax Normalize
 * For each row of the matrix, we do softmax normalization
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void matrix_softmax_normalize(float *C, size_t m, size_t n) {
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

/**
 * Vector to One-Hot Matrix
 * Transform a label vector y to the one-hot encoding matrix Y
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void vector_to_one_hot_matrix(const unsigned char *y, float *Y, size_t m,
                              size_t k) {
  // BEGIN YOUR CODE
  std::fill(Y, Y + m * k, 0.0f);

  // Set the appropriate element for each row to 1

  for (size_t i = 0; i < m; ++i) {
    size_t label = y[i];
    if (label < k) {
      Y[i * k + label] = 1.0f;
    }
  }
  // END YOUR CODE
}

/**
 * A C++ version of the softmax regression epoch code.  This should run a
 * single epoch over the data defined by X and y (and sizes m,n,k), and
 * modify theta in place.  Your function will probably want to allocate
 * (and then delete) some helper arrays to store the logits and gradients.
 *
 * Args:
 *     X (const float *): pointer to X data, of size m*n, stored in row
 *          major (C) format
 *     y (const unsigned char *): pointer to y data, of size m
 *     theta (float *): pointer to theta data, of size n*k, stored in row
 *          major (C) format
 *     m (size_t): number of examples
 *     n (size_t): input dimension
 *     k (size_t): number of classes
 *     lr (float): learning rate / SGD step size
 *     batch (int): SGD minibatch size
 *
 * Returns:
 *     (None)
 */
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
  // // BEGIN YOUR CODE
  for (size_t i = 0; i < m; i += batch) {
    size_t current_batch_size = std::min(batch, m - i);
    float *logits = new float[current_batch_size * k];
    float *Y = new float[current_batch_size * k];
    float *grad = new float[n * k];
    memset(grad, 0, n * k * sizeof(float));

    // Convert the label vector y to a one-hot encoded matrix Y for the current
    // batch
    vector_to_one_hot_matrix(y + i, Y, current_batch_size, k);

    // Forward pass: Compute logits for the current batch
    matrix_dot(X + i * n, theta, logits, current_batch_size, n, k);

    // Apply softmax normalization on logits
    matrix_softmax_normalize(logits, current_batch_size, k);

    // Compute gradient for the current batch
    float *logits_minus_Y = new float[current_batch_size * k];
    memcpy(logits_minus_Y, logits, current_batch_size * k * sizeof(float));
    matrix_minus(logits_minus_Y, Y, current_batch_size, k);
    matrix_dot_trans(X + i * n, logits_minus_Y, grad, current_batch_size, n, k);
    matrix_div_scalar(grad, static_cast<float>(current_batch_size), n, k);

    // Update theta: theta -= lr * grad
    matrix_mul_scalar(grad, lr, n, k);
    matrix_minus(theta, grad, n, k);

    delete[] logits;
    delete[] Y;
    delete[] grad;
    delete[] logits_minus_Y;
  }

  // int iterations = (m + batch - 1) / batch;
  // for (int iter = 0; iter < iterations; iter++) {
  //     const float *x = &X[iter * batch * n]; // x: batch x n
  //     float *Z = new float[batch * k];     // Z: batch x k
  //     matrix_dot(x, theta, Z, batch, n, k);
  //     for (int i = 0; i < batch * k; i++) Z[i] = exp(Z[i]); // element-wise
  //     exp for (int i = 0; i < batch; i++) {
  //         float sum = 0;
  //         for (int j = 0; j < k; j++) sum += Z[i * k + j];
  //         for (int j = 0; j < k; j++) Z[i * k + j] /= sum; // row-wise
  //         normalization
  //     }
  //     for (int i = 0; i < batch; i++) Z[i * k + y[iter * batch + i]] -= 1; //
  //     minus one-hot vector float *x_T = new float[n * batch]; float *grad =
  //     new float[n * k]; for (int i = 0; i < batch; i++)
  //         for (int j = 0; j < n; j++)
  //             x_T[j * batch + i] = x[i * n + j];
  //     matrix_dot(x_T, Z, grad, n, batch, k);
  //     for (int i = 0; i < n * k; i++) theta[i] -= lr / batch * grad[i]; //
  //     SGD update delete[] Z; delete[] x_T; delete[] grad;
  // }

  // END YOUR CODE
}

/**
 *Example function to fully train a softmax classifier
 **/
void train_softmax(const DataSet *train_data, const DataSet *test_data,
                   size_t num_classes, size_t epochs, float lr, size_t batch) {
  size_t size = train_data->input_dim * num_classes;
  float *theta = new float[size];
  memset(theta, 0, size * sizeof(float));
  float *train_result = new float[train_data->images_num * num_classes];
  float *test_result = new float[test_data->images_num * num_classes];
  float train_loss, train_err, test_loss, test_err;
  std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |"
            << std::endl;
  auto start_time = std::chrono::high_resolution_clock::now();
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    // BEGIN YOUR CODE
      softmax_regression_epoch_cpp(
          train_data->images_matrix,
          train_data->labels_array, theta, train_data->images_num,
          train_data->input_dim, num_classes, lr, batch);

    // After updating theta, compute predictions on train and test data
    matrix_dot(train_data->images_matrix, theta, train_result,
               train_data->images_num, train_data->input_dim, num_classes);
    matrix_dot(test_data->images_matrix, theta, test_result,
               test_data->images_num, test_data->input_dim, num_classes);

    // END YOUR CODE
    train_loss = mean_softmax_loss(train_result, train_data->labels_array,
                                   train_data->images_num, num_classes);
    test_loss = mean_softmax_loss(test_result, test_data->labels_array,
                                  test_data->images_num, num_classes);
    train_err = mean_err(train_result, train_data->labels_array,
                         train_data->images_num, num_classes);
    test_err = mean_err(test_result, test_data->labels_array,
                        test_data->images_num, num_classes);
    std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
              << std::fixed << std::setprecision(5) << train_loss << " |   "
              << std::fixed << std::setprecision(5) << train_err << " |   "
              << std::fixed << std::setprecision(5) << test_loss << " |  "
              << std::fixed << std::setprecision(5) << test_err << " |"
              << std::endl;
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
  delete[] theta;
  delete[] train_result;
  delete[] test_result;
}

/*
 *Return softmax loss.  Note that for the purposes of this assignment,
 *you don't need to worry about "nicely" scaling the numerical properties
 *of the log-sum-exp computation, but can just compute this directly.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average softmax loss over the sample.
 */
float mean_softmax_loss(const float *result, const unsigned char *labels_array,
                        size_t images_num, size_t num_classes) {
  // BEGIN YOUR CODE
  float loss = 0.0;
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

/*
 *Return error.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average error over the sample.
 */
float mean_err(float *result, const unsigned char *labels_array,
               size_t images_num, size_t num_classes) {
  // BEGIN YOUR CODE
  size_t incorrect_count = 0;
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

/**
 * Matrix Multiplication
 * Efficiently compute A = A * B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_mul(float *A, const float *B, size_t size) {
  // BEGIN YOUR CODE
  for (size_t i = 0; i < size; ++i) {
    A[i] *= B[i];
  }
  // END YOUR CODE
}

/*
Run a single epoch of SGD for a two-layer neural network defined by the
weights W1 and W2 (with no bias terms):
    logits = ReLU(X * W1) * W2
The function should use the step size lr, and the specified batch size (and
again, without randomizing the order of X).  It should modify the
W1 and W2 matrices in place.
Args:
    X: 1D input array of size
        (num_examples x input_dim).
    y: 1D class label array of size (num_examples,)
    W1: 1D array of first layer weights, of shape
        (input_dim x hidden_dim)
    W2: 1D array of second layer weights, of shape
        (hidden_dim x num_classes)
    m: num_examples
    n: input_dim
    l: hidden_dim
    k: num_classes
    lr (float): step size (learning rate) for SGD
    batch (int): size of SGD minibatch
*/
void nn_epoch_cpp(const float *X, const unsigned char *y, float *W1, float *W2,
                  size_t m, size_t n, size_t l, size_t k, float lr,
                  size_t batch) {
  // BEGIN YOUR CODE
  //   for (size_t i = 0; i < m; ++i) {
  //     // Forward pass
  //     float *hidden = new float[l];
  //     for (size_t j = 0; j < l; ++j) {
  //       hidden[j] = 0;
  //       for (size_t p = 0; p < n; ++p) {
  //         hidden[j] += X[i * n + p] * W1[p * l + j];
  //       }
  //       hidden[j] = std::max(0.0f, hidden[j]); // ReLU activation
  //     }

  //     float *output = new float[k];
  //     for (size_t j = 0; j < k; ++j) {
  //       output[j] = 0;
  //       for (size_t p = 0; p < l; ++p) {
  //         output[j] += hidden[p] * W2[p * k + j];
  //       }
  //     }

  //     // Backward pass and weight updates would go here

  //     delete[] hidden;
  //     delete[] output;
  //   }
  for (size_t i = 0; i < m; i += batch) {
    size_t current_batch_size = std::min(batch, m - i);

    float *Z1 = new float[current_batch_size * l];
    matrix_dot(X + i * n, W1, Z1, current_batch_size, n, l);
    for (size_t idx = 0; idx < current_batch_size * l; idx++) {
      if (Z1[idx] < 0) {
        Z1[idx] = 0.0;
      }
    }

    float *Z1_exp = new float[current_batch_size * k];
    matrix_dot(Z1, W2, Z1_exp, current_batch_size, l, k);

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
    vector_to_one_hot_matrix(y + i, Y, current_batch_size, k);

    float *Z2_minus_Y = new float[current_batch_size * k];
    memcpy(Z2_minus_Y, Z2, current_batch_size * k * sizeof(float));
    matrix_minus(Z2_minus_Y, Y, current_batch_size, k);

    float *G1 = new float[current_batch_size * l];
    matrix_trans_dot(Z2_minus_Y, W2, G1, current_batch_size, k, l);
    for (size_t idx = 0; idx < current_batch_size * l; ++idx) {
      G1[idx] = (Z1[idx] > 0) ? G1[idx] : 0;
    }

    float *W1_1 = new float[n * l];
    matrix_dot_trans(X + i * n, G1, W1_1, current_batch_size, n, l);
    matrix_div_scalar(W1_1, current_batch_size, n, l);
    matrix_mul_scalar(W1_1, lr, n, l);

    float *W2_1 = new float[l * k];
    matrix_dot_trans(Z1, Z2_minus_Y, W2_1, current_batch_size, l, k);
    matrix_div_scalar(W2_1, current_batch_size, l, k);
    matrix_mul_scalar(W2_1, lr, l, k);

    matrix_minus(W1, W1_1, n, l);
    matrix_minus(W2, W2_1, l, k);
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

/**
 *Example function to fully train a nn classifier
 **/
void train_nn(const DataSet *train_data, const DataSet *test_data,
              size_t num_classes, size_t hidden_dim, size_t epochs, float lr,
              size_t batch) {
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
  float *train_result = new float[train_data->images_num * num_classes];
  float *test_result = new float[test_data->images_num * num_classes];
  float train_loss, train_err, test_loss, test_err;
  std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |"
            << std::endl;
  auto start_time = std::chrono::high_resolution_clock::now();
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    // BEGIN YOUR CODE
    // for (size_t i = 0; i < train_data->images_num; i += batch) {
    // size_t current_batch_size = std::min(batch, train_data->images_num - i);
    nn_epoch_cpp(train_data->images_matrix, train_data->labels_array, W1, W2,
                 train_data->images_num, train_data->input_dim, hidden_dim,
                 num_classes, lr, batch);
    // }

    float *G1 = new float[train_data->images_num * hidden_dim];
    matrix_dot(train_data->images_matrix, W1, G1, train_data->images_num,
               train_data->input_dim, hidden_dim);
    apply_relu(G1, G1, train_data->images_num, hidden_dim);
    matrix_dot(G1, W2, train_result, train_data->images_num, hidden_dim,
               num_classes);

    float *G2 = new float[test_data->images_num * hidden_dim];
    matrix_dot(test_data->images_matrix, W1, G2, test_data->images_num,
               test_data->input_dim, hidden_dim);
    apply_relu(G2, G2, test_data->images_num, hidden_dim);
    matrix_dot(G2, W2, test_result, test_data->images_num, hidden_dim,
               num_classes);

    // END YOUR CODE
    train_loss = mean_softmax_loss(train_result, train_data->labels_array,
                                   train_data->images_num, num_classes);
    test_loss = mean_softmax_loss(test_result, test_data->labels_array,
                                  test_data->images_num, num_classes);
    train_err = mean_err(train_result, train_data->labels_array,
                         train_data->images_num, num_classes);
    test_err = mean_err(test_result, test_data->labels_array,
                        test_data->images_num, num_classes);
    std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
              << std::fixed << std::setprecision(5) << train_loss << " |   "
              << std::fixed << std::setprecision(5) << train_err << " |   "
              << std::fixed << std::setprecision(5) << test_loss << " |  "
              << std::fixed << std::setprecision(5) << test_err << " |"
              << std::endl;
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Execution Time: " << elapsed_time.count() << "milliseconds\n";
  delete[] W1;
  delete[] W2;
  delete[] train_result;
  delete[] test_result;
}
