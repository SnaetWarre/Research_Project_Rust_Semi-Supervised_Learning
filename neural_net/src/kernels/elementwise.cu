// CUDA Kernels for Element-wise Operations
// Optimized for maximum throughput on GPU

extern "C" {

// Element-wise addition: C = A + B
__global__ void add(const double* a, const double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Element-wise subtraction: C = A - B
__global__ void sub(const double* a, const double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

// Element-wise multiplication (Hadamard product): C = A * B
__global__ void mul(const double* a, const double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Element-wise division: C = A / B
__global__ void div(const double* a, const double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

// Scalar addition: C = A + scalar
__global__ void add_scalar(const double* a, double scalar, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + scalar;
    }
}

// Scalar multiplication: C = A * scalar
__global__ void mul_scalar(const double* a, double scalar, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * scalar;
    }
}

// In-place scalar multiplication: A *= scalar
__global__ void mul_scalar_inplace(double* a, double scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scalar;
    }
}

// Square each element: C = A^2
__global__ void square(const double* a, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = a[idx];
        c[idx] = val * val;
    }
}

// Square root: C = sqrt(A)
__global__ void sqrt_elementwise(const double* a, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = sqrt(a[idx]);
    }
}

// Exponential: C = exp(A)
__global__ void exp_elementwise(const double* a, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = exp(a[idx]);
    }
}

// Natural logarithm: C = log(A)
__global__ void log_elementwise(const double* a, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = log(a[idx]);
    }
}

// Power: C = A^power
__global__ void pow_elementwise(const double* a, double power, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = pow(a[idx], power);
    }
}

// Absolute value: C = |A|
__global__ void abs_elementwise(const double* a, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fabs(a[idx]);
    }
}

// Clamp values: C = clamp(A, min, max)
__global__ void clamp(const double* a, double min_val, double max_val, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = a[idx];
        c[idx] = fmin(fmax(val, min_val), max_val);
    }
}

// Broadcast addition: C[i,j] = A[i,j] + B[j] (add row vector to each row)
__global__ void add_row_vector(const double* a, const double* b, double* c, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int col = idx % cols;
        c[idx] = a[idx] + b[col];
    }
}

// Sum reduction along axis 0 (sum columns, output row vector)
// Each block handles one column
__global__ void sum_axis_0(const double* input, double* output, int rows, int cols) {
    int col = blockIdx.x;
    if (col >= cols) return;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    extern __shared__ double shared_sum[];
    
    // Each thread sums its subset of rows
    double thread_sum = 0.0;
    for (int row = tid; row < rows; row += stride) {
        thread_sum += input[row * cols + col];
    }
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[col] = shared_sum[0];
    }
}

// Mean reduction along axis 0
__global__ void mean_axis_0(const double* input, double* output, int rows, int cols) {
    int col = blockIdx.x;
    if (col >= cols) return;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    extern __shared__ double shared_sum[];
    
    double thread_sum = 0.0;
    for (int row = tid; row < rows; row += stride) {
        thread_sum += input[row * cols + col];
    }
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[col] = shared_sum[0] / rows;
    }
}

// Fill tensor with a constant value
__global__ void fill(double* data, double value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

// Copy tensor: dst = src
__global__ void copy(const double* src, double* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Transpose matrix (simple version, not optimized for memory coalescing)
__global__ void transpose(const double* input, double* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        output[col * rows + row] = input[row * cols + col];
    }
}

// Optimized transpose using shared memory (for better coalescing)
__global__ void transpose_optimized(const double* input, double* output, int rows, int cols) {
    __shared__ double tile[32][33]; // 33 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Load into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();
    
    // Transpose and write out
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Dropout forward: randomly zero out elements
__global__ void dropout_forward(const double* input, double* output, const double* mask, double scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * mask[idx] * scale;
    }
}

// Dropout backward: backpropagate through dropout
__global__ void dropout_backward(const double* grad_output, double* grad_input, const double* mask, double scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * mask[idx] * scale;
    }
}

// Generate dropout mask (0 or 1) based on keep probability
// Uses cuRAND state per thread
__global__ void generate_dropout_mask(double* mask, double keep_prob, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple LCG pseudo-random number generator
        unsigned long long state = seed + idx;
        state = (state * 6364136223846793005ULL + 1442695040888963407ULL);
        double rand_val = (double)(state >> 32) / 4294967296.0;
        mask[idx] = (rand_val < keep_prob) ? 1.0 : 0.0;
    }
}

} // extern "C"