// CUDA Kernels for Activation Functions
// Optimized for maximum performance on GPU

// Define DBL_MAX for CUDA kernels (NVRTC doesn't have access to system headers)
#ifndef DBL_MAX
#define DBL_MAX 1.7976931348623158e+308
#endif

extern "C" {

// ReLU activation: f(x) = max(0, x)
__global__ void relu_forward(const double* input, double* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmax(0.0, input[idx]);
    }
}

// ReLU backward: df/dx = 1 if x > 0, else 0
__global__ void relu_backward(const double* input, const double* grad_output, double* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = (input[idx] > 0.0) ? grad_output[idx] : 0.0;
    }
}

// Sigmoid activation: f(x) = 1 / (1 + exp(-x))
__global__ void sigmoid_forward(const double* input, double* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0 / (1.0 + exp(-input[idx]));
    }
}

// Sigmoid backward: df/dx = f(x) * (1 - f(x))
__global__ void sigmoid_backward(const double* output, const double* grad_output, double* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double sig = output[idx];
        grad_input[idx] = grad_output[idx] * sig * (1.0 - sig);
    }
}

// Tanh activation: f(x) = tanh(x)
__global__ void tanh_forward(const double* input, double* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanh(input[idx]);
    }
}

// Tanh backward: df/dx = 1 - f(x)^2
__global__ void tanh_backward(const double* output, const double* grad_output, double* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double t = output[idx];
        grad_input[idx] = grad_output[idx] * (1.0 - t * t);
    }
}

// Leaky ReLU activation: f(x) = x if x > 0, else alpha * x
__global__ void leaky_relu_forward(const double* input, double* output, double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] > 0.0) ? input[idx] : alpha * input[idx];
    }
}

// Leaky ReLU backward
__global__ void leaky_relu_backward(const double* input, const double* grad_output, double* grad_input, double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = (input[idx] > 0.0) ? grad_output[idx] : alpha * grad_output[idx];
    }
}

// Softmax forward (per row for batched data)
// Each block processes one row
__global__ void softmax_forward(const double* input, double* output, int batch_size, int num_classes) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Shared memory for reduction
    extern __shared__ double shared[];
    double* max_vals = shared;
    double* sum_vals = &shared[blockDim.x];
    
    // Find max value in row (for numerical stability)
    double thread_max = -DBL_MAX;
    for (int i = tid; i < num_classes; i += stride) {
        thread_max = fmax(thread_max, input[row * num_classes + i]);
    }
    max_vals[tid] = thread_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            max_vals[tid] = fmax(max_vals[tid], max_vals[tid + s]);
        }
        __syncthreads();
    }
    double row_max = max_vals[0];
    __syncthreads();
    
    // Compute exp(x - max) and sum
    double thread_sum = 0.0;
    for (int i = tid; i < num_classes; i += stride) {
        double val = exp(input[row * num_classes + i] - row_max);
        output[row * num_classes + i] = val;
        thread_sum += val;
    }
    sum_vals[tid] = thread_sum;
    __syncthreads();
    
    // Reduce to find global sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_vals[tid] += sum_vals[tid + s];
        }
        __syncthreads();
    }
    double row_sum = sum_vals[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < num_classes; i += stride) {
        output[row * num_classes + i] /= row_sum;
    }
}

// Softmax backward (combined with cross-entropy for efficiency)
// grad = output - target (one-hot)
__global__ void softmax_cross_entropy_backward(const double* output, const double* target, double* grad, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_classes;
    
    if (idx < total) {
        grad[idx] = output[idx] - target[idx];
    }
}

} // extern "C"