// GPU data preparation kernels for CIFAR-10
// - Convert interleaved HWC u8 RGB to CHW double
// - Normalize to [0,1]
// - Optional horizontal flip and random crop with zero padding

extern "C" {

// in_hwc:  (batch, H*W*3) interleaved RGB (u8)
// out_chw: (batch, 3*OH*OW) channels-first (double)
// flip:    (batch) int flags: 0 no flip, 1 horizontal flip
// offsets: (batch*2) int (dy, dx) crop offsets relative to original image (can be negative)
// H, W are input dims (32,32), OH, OW are output dims (typically 32,32). pad is implicit via bounds check (zeros outside)
__global__ void hwc_to_chw_norm_augment(
    const unsigned char* __restrict__ in_hwc,
    double* __restrict__ out_chw,
    const int* __restrict__ flip,
    const int* __restrict__ offsets,
    int batch,
    int H,
    int W,
    int OH,
    int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * 3 * OH * OW;
    if (idx >= total) return;

    int n = idx / (3 * OH * OW);
    int rem = idx % (3 * OH * OW);
    int c = rem / (OH * OW);           // 0..2
    int pix = rem % (OH * OW);
    int oy = pix / OW;
    int ox = pix % OW;

    // crop offsets per image
    int dy = offsets[2*n + 0];
    int dx = offsets[2*n + 1];

    // map output (oy,ox) to source (iy,ix)
    int ix = ox + dx;
    int iy = oy + dy;

    // horizontal flip if requested (flip around image center)
    if (flip[n] != 0) {
        ix = (OW - 1 - ox) + dx; // flip before offset
    }

    double val = 0.0;
    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
        int hw_idx = iy * W + ix;
        int src = hw_idx * 3 + c; // HWC interleaved RGB
        unsigned char bytev = in_hwc[n * (H*W*3) + src];
        val = ((double)bytev) / 255.0;
    }

    int dst = n * (3 * OH * OW) + c * (OH * OW) + oy * OW + ox;
    out_chw[dst] = val;
}

// Cutout: zero out a square region per image on CHW-normalized tensor
// inputs/out: (batch, 3*H*W) double; cutout: (batch*3) int (cy, cx, half)
__global__ void cutout_apply(
    double* __restrict__ chw,
    const int* __restrict__ cutout,
    int batch,
    int H,
    int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * 3 * H * W;
    if (idx >= total) return;

    int n = idx / (3 * H * W);
    int rem = idx % (3 * H * W);
    int c = rem / (H * W);
    int pix = rem % (H * W);
    int y = pix / W;
    int x = pix % W;

    int cy = cutout[3*n + 0];
    int cx = cutout[3*n + 1];
    int half = cutout[3*n + 2];

    if (x >= cx - half && x <= cx + half && y >= cy - half && y <= cy + half) {
        int dst = n * (3 * H * W) + c * (H * W) + y * W + x;
        chw[dst] = 0.0;
    }
}

// Color jitter: brightness and contrast per image
// params: (batch*2) double: brightness in [-b,b], contrast in [1-c,1+c]
__global__ void color_jitter(
    double* __restrict__ chw,
    const double* __restrict__ params,
    int batch,
    int H,
    int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * 3 * H * W;
    if (idx >= total) return;

    int n = idx / (3 * H * W);
    int rem = idx % (3 * H * W);
    int c = rem / (H * W);
    int pix = rem % (H * W);
    int y = pix / W;
    int x = pix % W;

    double b = params[2*n + 0];
    double k = params[2*n + 1];
    int dst = n * (3 * H * W) + c * (H * W) + y * W + x;
    double v = chw[dst];
    // clamp to [0,1]
    v = k * v + b;
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    chw[dst] = v;
}

}


