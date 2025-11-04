// Matrix operations CUDA kernels
// Using double precision to match Float = f64

// Define DBL_MAX for CUDA kernels (NVRTC doesn't have access to system headers)
#ifndef DBL_MAX
#define DBL_MAX 1.7976931348623158e+308
#endif

// Optimized matrix transpose using shared memory
extern "C" __global__ void transpose(const double* input, double* output, int rows, int cols) {
    __shared__ double tile[16][17]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Load into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // Write transposed output
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Sum along rows (reduce columns)
extern "C" __global__ void sum_rows(const double* input, double* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        double sum = 0.0;
        for (int col = 0; col < cols; col++) {
            sum += input[row * cols + col];
        }
        output[row] = sum;
    }
}

// Sum along columns (reduce rows) - result is a row vector
extern "C" __global__ void sum_cols(const double* input, double* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {
        double sum = 0.0;
        for (int row = 0; row < rows; row++) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

// Broadcast add row vector to each row of matrix
extern "C" __global__ void add_row_vector(const double* matrix, const double* vector, double* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] = matrix[idx] + vector[col];
    }
}

// Mean along rows
extern "C" __global__ void mean_rows(const double* input, double* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        double sum = 0.0;
        for (int col = 0; col < cols; col++) {
            sum += input[row * cols + col];
        }
        output[row] = sum / (double)cols;
    }
}

// Mean along columns
extern "C" __global__ void mean_cols(const double* input, double* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {
        double sum = 0.0;
        for (int row = 0; row < rows; row++) {
            sum += input[row * cols + col];
        }
        output[col] = sum / (double)rows;
    }
}

// Variance along columns (for batch norm)
extern "C" __global__ void variance_cols(const double* input, const double* mean, double* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {
        double sum_sq_diff = 0.0;
        double m = mean[col];
        
        for (int row = 0; row < rows; row++) {
            double diff = input[row * cols + col] - m;
            sum_sq_diff += diff * diff;
        }
        
        output[col] = sum_sq_diff / (double)rows;
    }
}

// Normalize columns (subtract mean, divide by std)
extern "C" __global__ void normalize_cols(const double* input, const double* mean, const double* std, double* output, int rows, int cols, double epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rows * cols) {
        int col = idx % cols;
        double m = mean[col];
        double s = std[col];
        output[idx] = (input[idx] - m) / (s + epsilon);
    }
}

// Scale and shift (for batch norm after normalization)
extern "C" __global__ void scale_shift(const double* input, const double* scale, const double* shift, double* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] = input[idx] * scale[col] + shift[col];
    }
}

// Dropout kernel
extern "C" __global__ void dropout(const double* input, const double* mask, double scale, double* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = input[idx] * mask[idx] * scale;
    }
}

// Generate dropout mask
extern "C" __global__ void generate_dropout_mask(double* mask, double keep_prob, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Simple random number generator (not cryptographically secure)
        unsigned long long state = seed + idx;
        state = state * 6364136223846793005ULL + 1;
        double rand = (double)(state >> 32) / (double)0xFFFFFFFF;
        mask[idx] = rand < keep_prob ? 1.0 : 0.0;
    }
}

// Matrix-matrix element-wise multiplication with broadcasting
extern "C" __global__ void mul_broadcast(const double* a, const double* b, double* output, int rows, int cols, int b_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;
        
        // If b has only 1 row, broadcast across all rows
        int b_idx = (b_rows == 1) ? col : (row * cols + col);
        output[idx] = a[idx] * b[b_idx];
    }
}

// Fill matrix with value
extern "C" __global__ void fill(double* output, double value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = value;
    }
}

// Copy matrix
extern "C" __global__ void copy(const double* input, double* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = input[idx];
    }
}

// Max reduction along rows
extern "C" __global__ void max_rows(const double* input, double* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        double max_val = -DBL_MAX;
        for (int col = 0; col < cols; col++) {
            max_val = fmax(max_val, input[row * cols + col]);
        }
        output[row] = max_val;
    }
}

// Argmax along rows
extern "C" __global__ void argmax_rows(const double* input, int* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        double max_val = -DBL_MAX;
        int max_idx = 0;
        
        for (int col = 0; col < cols; col++) {
            double val = input[row * cols + col];
            if (val > max_val) {
                max_val = val;
                max_idx = col;
            }
        }
        
        output[row] = max_idx;
    }
}