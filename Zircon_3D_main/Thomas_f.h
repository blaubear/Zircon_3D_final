

__global__
void Thomas_to_b(double* a, double* b, double* c, int NX, int  NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    b[i + NX * (j + 1)] = -c[i + NX * (j+1)]/((b[i + NX * (j + 1)])+a[i + NX * (j + 1)] * b[i + NX * (j )]);

}

__global__
void Thomas_to_rhs(double* a, double* b, double* c, double* rhs, int NX, int  NY, double epsilon, int line_num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((c[i + NX * (j + 1)] < epsilon) && (c[i + NX * (j + 1)] > -epsilon) && (line_num <= NY-5) ) {
        //rhs[i + NY * (j + 1)] = (rhs[i + NY * (j + 1)] - a[i + NY * (j + 1)] * rhs[i + NY * (j)]) / ((b[i + NX * (j + 1)]) + a[i + NX * (j + 1)] * b[i + NX * (j)]);
        //rhs[i + NY * (j + 1)] = rhs[i + NY * (j + 1)];
    }
    else {
        rhs[i + NY * (j + 1)] = (rhs[i + NY * (j + 1)] - a[i + NY * (j + 1)] * rhs[i + NY * (j)]) / ((b[i + NX * (j + 1)]) + a[i + NX * (j + 1)] * b[i + NX * (j)]);
    }
}

__global__
void Thomas_delenie(double* d_C, double* b, double* rhs, int NX, int  NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    d_C[i + (NY)*j] = rhs[i + (NY)*j];
}

__global__
void Thomas_back(double* d_C, double* a, double* b, double* c, double* rhs, int NX, int  NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    d_C[i + (NY - 2) * j] = rhs[i + 1 + NY * j] + d_C[i + (NY - 2) * (j + 1)] * b[i + 1 + NY * j];
}

__global__
void Thomas_back_rev(double* d_C, double* a, double* b, double* c, double* rhs, int NX, int  NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    d_C[i + (NY - 2) * j] = (rhs[i + 1 + NY * j] - c[i + 1 + NY * j] * d_C[i + (NY - 2) * (j + 1)]) / b[i + 1 + NY * j];
}


void Thomas(double* d_C, int  NX, int  NY, double* a, double* b, double* c, double* Rhs, double* x, int Rev, int SIZE_OF_BLOCK) {
    dim3 THREADS_SIZE_L(SIZE_OF_BLOCK, 1);
    dim3 BLOCK_SIZE_L((NX + THREADS_SIZE_L.x - 1) / THREADS_SIZE_L.x, 1);
    double epsilon = 1.e-10;
    //dim3 THREADS_SIZE(SIZE_OF_BLOCK, SIZE_OF_BLOCK);
    //dim3 BLOCK_SIZE((NX + THREADS_SIZE.x - 1) / THREADS_SIZE.x, (NY + THREADS_SIZE.y - 1) / THREADS_SIZE.y);


    for (int i = 1; i <= NY + 2; i++) {
        Thomas_to_rhs << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&a[(NX + 2) * (i - 1)], &b[(NX + 2) * (i - 1)], &c[(NX + 2) * (i - 1)], &Rhs[(NX + 2) * (i - 1)], (NX + 2), (NX + 2), epsilon, i);
        Thomas_to_b << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&a[(NX + 2) * (i - 1)], &b[(NX + 2) * (i - 1)], &c[(NX + 2) * (i - 1)], NX + 2, NY + 2);
    }

    Thomas_delenie << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&x[(NX) * (NY + 1)], &b[(NX + 2) * (NY + 2 - 1)], &Rhs[(NX + 2) * (NY + 2 - 1)], NX + 2, NY + 2);
    //x(:, ny) = rhs(:, ny). / b(:, ny);
    for (int i = NY + 1; i >= 0; i--) {
            Thomas_back << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&x[(NX)*i], &a[(NX + 2) * i], &b[(NX + 2) * i], &c[(NX + 2) * i], &Rhs[(NX + 2) * i], NX + 2, NY + 2);
            //x(:, i) = (rhs(:, i) - c(:, i).*x(:, i + 1)). / b(:, i);
    }
}