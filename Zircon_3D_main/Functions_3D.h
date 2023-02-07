inline void gpuAssert_3D(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__
void Yavno_diffusion_3D(double* C, double* C_old, double D, double dx, double dy, double dz, double* Qx, double* Qy, double* Qz, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > 0 && i < (NX - 1) && j > 0 && j < (NY - 1) && k > 0 && k < (NZ - 1)) {
        C[i + NX * j + NX * NY * k] = C_old[i + NX * j + NX * NY * k] + dt * D * (Qx[i - 1 + NX * (j) + NX * NY * k] / dx / dx + Qy[i + NX * (j - 1) + NX * NY * k] / dy / dy + Qz[i + NX * j + NX * NY * (k - 1)] / dz / dz);
    }
}



__global__
void bounday_cond_3D(double* d_C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    d_C[NX - 1 + i * NX + NX * NY * j]      = d_C[NX - 2 + i * NX + NX * NY * j];
    d_C[i * NX + NX * NY * j]               = d_C[1 + i * NX + NX * NY * j];
    d_C[i + NX * (NY - 1) + NX * NY * j]    = d_C[i + NX * (NY - 2) + NX * NY * j];
    d_C[i + NX * NY * j]                    = d_C[i + NX + NX * NY * j];
    d_C[i + NX * j + NX * NY * (NZ - 1)]    = d_C[i + NX + NX * j + NX * NY * (NZ - 2)];
    d_C[i + NX * j]                         = d_C[i + NX + NX * j + NX * NY];
}

__global__
void bounday_cond_up_x_3D(double* d_C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    d_C[ NX - 1 + i * NX + NX * NY * j] = d_C[NX - 2 + i * NX + NX * NY * j];
}

__global__
void bounday_cond_back_x_3D(double* d_C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    d_C[i * NX + NX * NY * j] = d_C[1 + i * NX + NX * NY * j];
}

__global__
void bounday_cond_up_y_3D(double* d_C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    d_C[i + NX * (NY-1) + NX * NY * j] = d_C[i + NX * (NY - 2) + NX * NY * j];
}

__global__
void bounday_cond_back_y_3D(double* d_C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    d_C[i  + NX * NY * j] = d_C[i + NX + NX * NY * j];
}   
__global__
void bounday_cond_up_z_3D(double* d_C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    d_C[i + NX * j + NX * NY * (NZ-1)] = d_C[i + NX + NX * j + NX * NY * (NZ - 2)];
}

__global__
void bounday_cond_back_z_3D(double* d_C) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_C[i + NX * j] = d_C[i + NX + NX * j + NX * NY];
}

__global__
void Rhs_GPU_diff_x_3D(double* Rhs, double* C_old, int NX, int NY, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //Rhs[i + NY * j] = C_old[j + NY * i] - dt/2* divQ[j + NY * i];
    Rhs[i + 1 + NX * (j + 1)] = C_old[j + (NX - 2) * i] ;
}

__global__
void Rhs_GPU_diff_y_3D(double* Rhs, double* C_old, double* divQ, double* g, int NX, int NY, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //Rhs[i + NY * j] = C_old[i + NY * j] - dt / 2 * divQ[i + NY * j];
    Rhs[i + 1 + NX * (j + 1)] = C_old[i + (NX - 2) * j] ;
}

__global__
void Rhs_GPU_diff_z_3D(double* Rhs, double* C_old, double* divQ, double* g, int NX, int NY, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //Rhs[i + NY * j] = C_old[i + NY * j] - dt / 2 * divQ[i + NY * j];
    Rhs[i + 1 + NX * (j + 1)] = C_old[(NX - 2) * i + (NX - 2) * (NY - 2) * j] ;
}

__global__
void diff_func(double* qx, double* qy, double* qz, double* C_old) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < NX - 1) {
        qx[i + NX * j + NX * NY * k] = C_old[i + 1 + NX * j + NX * NY * k] - C_old[i + NX * j + NX * NY * k];
    }
    if (j < NY - 1) {
        qy[i + NX * j + NX * NY * k] = C_old[i + NX * (j + 1) + NX * NY * k] - C_old[i + NX * j + NX * NY * k];
    }
    if (k < NZ - 1) {
        qz[i + NX * j + NX * NY * k] = C_old[i + NX * j + NX * NY * (k + 1)] - C_old[i + NX * j + NX * NY * k];
    }
}

__global__
void diff_func_2(double* qx, double* qy, double* qz, double* C_old_x, double* C_old_y, double* C_old_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < NX - 1) {
        qx[i + NX * j + NX * NY * k] = C_old_x[i + 1 + NX * j + NX * NY * k] - C_old_x[i + NX * j + NX * NY * k];
    }
    if (j < NY - 1) {
        qy[i + NX * j + NX * NY * k] = C_old_y[i + NX * (j + 1) + NX * NY * k] - C_old_y[i + NX * j + NX * NY * k];
    }
    if (k < NZ - 1) {
        qz[i + NX * j + NX * NY * k] = C_old_z[i + NX * j + NX * NY * (k + 1)] - C_old_z[i + NX * j + NX * NY * k];
    }
}

__global__
void diff_x(double* Rhs, double* C_old) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < NX - 1) {
        Rhs[i + NX * j + NX * NY * k] = C_old[i + 1 + NX * j + NX * NY * k] - C_old[i + NX * j + NX * NY * k];
    }
}

__global__
void diff_y(double* Rhs, double* C_old) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (j < NY - 1) {
        Rhs[i + NX * j + NX * NY * k] = C_old[i + NX * (j + 1) + NX * NY * k] - C_old[i + NX * j + NX * NY * k];
    }
}

__global__
void diff_z(double* Rhs, double* C_old) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (k < NZ - 1) {
        Rhs[i + NX * j + NX * NY * k] = C_old[i  + NX * j + NX * NY * (k + 1)] - C_old[i + NX * j + NX * NY * k];
    }
}

__global__
void Coef_eq_3D(double* Rhs, int NX, int NY, double eq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Rhs[i + NY * j] = eq;
}

__global__
void Coef_eq_2end_3D(double* Rhs, int NX, int NY, double eq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Rhs[i + 1 + NY * (j + 1)] = eq;
}

__global__
void Coef_eq_2end_2end_3D(double* Rhs, int NX, int NY, double eq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Rhs[i + 1 + NY * (j)] = eq;
}

__global__
void Copy_3D(double* T, double* x, int NX, int NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    T[i + NX * (j)] = x[i + NX * (j + 1)];
}

__global__
void Copy_rev_3D(double* T, double* x, int NX, int NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    T[i + NX * (j)] = x[j * NX + NX * NY * (i + 1)];
}

__global__
void Copy_z_3D(double* T, double* x, int NX, int NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    T[NX * NY * i  +  NX * j] = x[NX * (i+1) + j];
}



/******************************/
/*      INITIALIZATION        */
/******************************/

void Initialize_3D(double* h_T, const int NX, const int NY, const int NZ, double value)
{
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                h_T[NX * NY * k + NY *j + i] = value;
            }
        }
    }
}

void Initialize_3D_test(double* h_T, const int NX, const int NY, const int NZ, double value)
{
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                h_T[NX * NY * k + NY * j + i] = value*i*j;
            }
        }
    }
}

__global__
void Coef_eq_addition_3D(double* Rhs, int NX, int NY, double eq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Rhs[i + 1 + NY * (j + 1)] = Rhs[i + 1 + NY * (j + 1)] - eq;
}

__global__
void distance_counter_3D(double* distance, int NX, int NY, double eq, int N_cryst, double* S_cent_x, double* S_cent_y) {
    int i_k = blockIdx.x * blockDim.x + threadIdx.x;
    int j_k = blockIdx.y * blockDim.y + threadIdx.y;
    distance[i_k + j_k * NX] = 0;
    //for (int N_count = 0; N_count <= N_cryst; N_count++) {
   //     distance[i_k + j_k * NX] = distance[i_k + j_k * NX] + pow(pow(fabs((i_k / (double)NX) - S_cent_x[N_count]), 1.0) + pow(fabs((j_k / (double)NY) - S_cent_y[N_count]), 1.0), 0.1);
    //}
}

__global__
void create_crystal_3D(double S_x, double S_y, double S_z, ALL_PARAMS_TYPES) {
    S_cent_x[N_cryst]     = S_x;
    S_cent_y[N_cryst]     = S_y;
    S_cent_z[N_cryst]     = S_z;

    S_left[N_cryst]       = S_x - S_0;
    S_right[N_cryst]      = S_x + S_0;
    S_bot[N_cryst]        = S_y - S_0;
    S_top[N_cryst]        = S_y + S_0;
    S_back[N_cryst]       = S_z - S_0;
    S_forth[N_cryst]      = S_z + S_0;

    S_cent_x_num[N_cryst] = (int)(S_x * (NX - 1));
    S_cent_y_num[N_cryst] = (int)(S_y * (NY - 1));
    S_cent_z_num[N_cryst] = (int)(S_z * (NZ - 1));

    Cryst_age[N_cryst]    = time;
}


double find_max_3D(double* array, int NX, int NY, int gap, double* answer) {
    double max = array[gap];
    answer[0] = gap;
    answer[1] = gap;
    //const int con = NX * NY;
    for (int i = gap; i < NX - gap; i++) {
        for (int j = gap; j < NY - gap; j++) {
            if (array[i + NX * j] > max) {
                max = array[i + NX * j];
                answer[0] = i;
                answer[1] = j;
            }
        }
    }
    return max;
}


__global__
void k_refresh_3D(ALL_PARAMS_TYPES) {
        int i_k = blockIdx.x * blockDim.x + threadIdx.x;

        k_left_arr[i_k] = S_cent_x_num[i_k];
        k_right_arr[i_k] = S_cent_x_num[i_k];
        k_top_arr[i_k] = S_cent_y_num[i_k];
        k_bot_arr[i_k] = S_cent_y_num[i_k];
        k_back_arr[i_k] = S_cent_z_num[i_k];
        k_forth_arr[i_k] = S_cent_z_num[i_k];

        while (k_left_arr[i_k] * dx >= S_left[i_k] && k_left_arr[i_k] < NX) {
            k_left_arr[i_k]--;
        }
        k_left_arr[i_k]++;


        while (k_right_arr[i_k] * dx <= S_right[i_k] && k_right_arr[i_k] < NX) {
            k_right_arr[i_k]++;
        }


        while (k_bot_arr[i_k] * dx >= S_bot[i_k] && k_bot_arr[i_k] < NX) {
            k_bot_arr[i_k]--;
        }
        k_bot_arr[i_k]++;


        while (k_top_arr[i_k] * dx <= S_top[i_k] && k_top_arr[i_k] < NX) {
            k_top_arr[i_k]++;
        }

        while (k_back_arr[i_k] * dx >= S_back[i_k] && k_back_arr[i_k] < NX) {
            k_back_arr[i_k]--;
        }
        k_back_arr[i_k]++;


        while (k_forth_arr[i_k] * dx <= S_forth[i_k] && k_forth_arr[i_k] < NX) {
            k_forth_arr[i_k]++;
        }

}

__global__
void S_bot_refresh_3D(ALL_PARAMS_TYPES) {

    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    summ[0] = 0;
    for (int N_count_in = 0; N_count_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_forth_arr[i_k] - k_back_arr[i_k]; N_count_in_in++) {
            summ[0] = summ[0] + d_C[NX * NY * (k_back_arr[i_k] + N_count_in_in) + NX * (k_bot_arr[i_k] - 1) + k_left_arr[i_k] + N_count_in];
        }
    }

    summ[1] = 0;
    for (int N_count_in = 0; N_count_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_forth_arr[i_k] - k_back_arr[i_k]; N_count_in_in++) {
            summ[1] = summ[1] + d_C[NX * NY * (k_back_arr[i_k] + N_count_in_in) + NX * (k_bot_arr[i_k]) + k_left_arr[i_k] + N_count_in];
        }
    }

    dCpoDx_bot[i_k] = -(summ[0] - summ[1]) / ((k_right_arr[i_k] - k_left_arr[i_k]) * (k_forth_arr[i_k] - k_back_arr[i_k]) * dx);

    V_bot[i_k] = -D_nd * dCpoDx_bot[i_k] / (C_sat - C_cryst);
    if (FLAG_V_limit){
        if (V_bot[i_k] < -a * pow((D_nd / dt / 3), (0.5)) || V_bot[i_k] > 0) {
            V_bot[i_k] = -a * pow((D_nd / dt / 3), (0.5));
            dCpoDx_bot[i_k] = -V_bot[i_k] * (C_sat - C_cryst) / D_nd;
            FLAGGG_counter[0] = FLAGGG_counter[0] + 1;
        }
    }
}


__global__
void S_top_refresh_3D(ALL_PARAMS_TYPES) {

    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    summ[0] = 0;
    for (int N_count_in = 0; N_count_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_forth_arr[i_k] - k_back_arr[i_k]; N_count_in_in++) {
            summ[0] = summ[0] + d_C[NX * NY * (k_back_arr[i_k] + N_count_in_in) + NX * (k_top_arr[i_k] - 1) + k_left_arr[i_k] + N_count_in];
        }
    }

    summ[1] = 0;
    for (int N_count_in = 0; N_count_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_forth_arr[i_k] - k_back_arr[i_k]; N_count_in_in++) {
            summ[1] = summ[1] + d_C[NX * NY * (k_back_arr[i_k] + N_count_in_in) + NX * (k_top_arr[i_k]) + k_left_arr[i_k] + N_count_in];
        }
    }

    dCpoDx_top[i_k] = -(summ[0] - summ[1]) / ((k_right_arr[i_k] - k_left_arr[i_k]) * (k_forth_arr[i_k] - k_back_arr[i_k]) * dx);

    V_top[i_k] = -D_nd * dCpoDx_top[i_k] / (C_sat - C_cryst);
    if (FLAG_V_limit){
        if (V_top[i_k] > a * pow((D_nd / dt / 3), (0.5)) || V_top[i_k] < 0) {
            V_top[i_k] = a * pow((D_nd / dt / 3), (0.5));
            dCpoDx_top[i_k] = -V_top[i_k] * (C_sat - C_cryst) / D_nd;
            FLAGGG_counter[0] = FLAGGG_counter[0] + 1;
        }
    }
}


__global__
void S_left_refresh_3D(ALL_PARAMS_TYPES) {

    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    summ[0] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_forth_arr[i_k] - k_back_arr[i_k]; N_count_in_in++) {
            summ[0] = summ[0] + d_C[NX * NY * (k_back_arr[i_k] + N_count_in_in) + NX * (k_bot_arr[i_k] + N_count_in) + k_left_arr[i_k] - 1];
        }
    }

    summ[1] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_forth_arr[i_k] - k_back_arr[i_k]; N_count_in_in++) {
            summ[1] = summ[1] + d_C[NX * NY * (k_back_arr[i_k] + N_count_in_in) + NX * (k_bot_arr[i_k] + N_count_in) + k_left_arr[i_k]];
        }
    }

    dCpoDx_left[i_k] = -(summ[0] - summ[1]) / ((k_top_arr[i_k] - k_bot_arr[i_k]) * (k_forth_arr[i_k] - k_back_arr[i_k]) * dx);

    V_left[i_k] = -D_nd * dCpoDx_left[i_k] / (C_sat - C_cryst);
    if (FLAG_V_limit){
        if (V_left[i_k] < -a * pow((D_nd / dt /3), (0.5)) || V_left[i_k] > 0) {
            V_left[i_k] = -a * pow((D_nd / dt /3), (0.5));
            dCpoDx_left[i_k] = -V_left[i_k] * (C_sat - C_cryst) / D_nd;
            FLAGGG_counter[0] = FLAGGG_counter[0] + 1;
        }
    }
}


__global__
void S_right_refresh_3D(ALL_PARAMS_TYPES) {

    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    summ[0] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_forth_arr[i_k] - k_back_arr[i_k]; N_count_in_in++) {
            summ[0] = summ[0] + d_C[NX * NY * (k_back_arr[i_k] + N_count_in_in) + NX * (k_bot_arr[i_k] + N_count_in) + k_right_arr[i_k] - 1];
        }
    }

    summ[1] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_forth_arr[i_k] - k_back_arr[i_k]; N_count_in_in++) {
            summ[1] = summ[1] + d_C[NX * NY * (k_back_arr[i_k] + N_count_in_in) + NX * (k_bot_arr[i_k] + N_count_in) + k_right_arr[i_k]];
        }
    }

    dCpoDx_right[i_k] = -(summ[0] - summ[1]) / ((k_top_arr[i_k] - k_bot_arr[i_k]) * (k_forth_arr[i_k] - k_back_arr[i_k]) * dx);

    V_right[i_k] = -D_nd * dCpoDx_right[i_k] / (C_sat - C_cryst);

    if (FLAG_V_limit){
        if (V_right[i_k] > a * pow((D_nd / dt / 3), (0.5)) || V_right[i_k] < 0) {
            V_right[i_k] = a * pow((D_nd / dt / 3), (0.5));
            dCpoDx_right[i_k] = -V_right[i_k] * (C_sat - C_cryst) / D_nd;
            FLAGGG_counter[0] = FLAGGG_counter[0] + 1;
        }
    } 
}

__global__
void S_forth_refresh_3D(ALL_PARAMS_TYPES) {

    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    summ[0] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in_in++) {
            summ[0] = summ[0] + d_C[NX * NY * (k_forth_arr[i_k] - 1) + NX * (k_bot_arr[i_k] + N_count_in) + k_left_arr[i_k] + N_count_in_in];
        }
    }

    summ[1] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in_in++) {
            summ[1] = summ[1] + d_C[NX * NY * (k_forth_arr[i_k]    ) + NX * (k_bot_arr[i_k] + N_count_in) + k_left_arr[i_k] + N_count_in_in];
        }
    }

    dCpoDx_forth[i_k] = -(summ[0] - summ[1]) / ((k_top_arr[i_k] - k_bot_arr[i_k]) * (k_right_arr[i_k] - k_left_arr[i_k]) * dx);

    V_forth[i_k] = -D_nd * dCpoDx_forth[i_k] / (C_sat - C_cryst);

    if (FLAG_V_limit) {
        if (V_forth[i_k] > a * pow((D_nd / dt / 3), (0.5)) || V_forth[i_k] < 0) {
            V_forth[i_k] = a * pow((D_nd / dt / 3), (0.5));
            dCpoDx_forth[i_k] = -V_forth[i_k] * (C_sat - C_cryst) / D_nd;
            FLAGGG_counter[0] = FLAGGG_counter[0] + 1;
        }
    }
}

__global__
void S_back_refresh_3D(ALL_PARAMS_TYPES) {

    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    summ[0] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in_in++) {
            summ[0] = summ[0] + d_C[NX * NY * (k_back_arr[i_k] - 1 ) + NX * (k_bot_arr[i_k] + N_count_in) + k_left_arr[i_k] + N_count_in_in];
        }
    }

    summ[1] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        for (int N_count_in_in = 0; N_count_in_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in_in++) {
            summ[1] = summ[1] + d_C[NX * NY * (k_back_arr[i_k]      ) + NX * (k_bot_arr[i_k] + N_count_in) + k_left_arr[i_k] + N_count_in_in];
        }
    }

    dCpoDx_back[i_k] = -(summ[0] - summ[1]) / ((k_top_arr[i_k] - k_bot_arr[i_k]) * (k_right_arr[i_k] - k_left_arr[i_k]) * dx);

    V_back[i_k] = -D_nd * dCpoDx_back[i_k] / (C_sat - C_cryst);

    if (FLAG_V_limit) {
        if (V_back[i_k] < -a * pow((D_nd / dt / 3), (0.5)) || V_back[i_k] > 0) {
            V_back[i_k] = -a * pow((D_nd / dt / 3), (0.5));
            dCpoDx_back[i_k] = -V_back[i_k] * (C_sat - C_cryst) / D_nd;
            FLAGGG_counter[0] = FLAGGG_counter[0] + 1;
        }
    }
}


__global__
void Source_eq_x_3D(double* d_C, int NX, int NY, double C_sat, double dx, double D_nd, double* dCpoDx_left, double* dCpoDx_right, double* S_right, double* S_left, double* S_bot, double* S_top, double* dCpoDx_top, double* dCpoDx_bot, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int* k_forth_arr, int* k_back_arr) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < k_right_arr[ix] - k_left_arr[ix]; i++) {
        for (int j = 0; j < k_top_arr[ix] - k_bot_arr[ix]; j++) {
            for (int k = 0; k < k_forth_arr[ix] - k_back_arr[ix]; k++) {
                d_C[NX * NY * (k_back_arr[ix] + k) + NX * (k_bot_arr[ix] + j) + k_left_arr[ix] + i ] = ((-S_left[ix] * dCpoDx_left[ix] + (k_left_arr[ix] + i) * dx * dCpoDx_left[ix] + C_sat) * pow(S_right[ix], 2) + (-pow(S_left[ix], 2) * dCpoDx_right[ix] + ((2 * dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * (k_left_arr[ix] + i) * dx - 2 * C_sat) * S_left[ix] - (2. * (dCpoDx_left[ix] + dCpoDx_right[ix] / (2.))) * pow(k_left_arr[ix] + i, 2) * pow(dx, 2)) * S_right[ix] + ((k_left_arr[ix] + i) * dx * dCpoDx_right[ix] + C_sat) * pow(S_left[ix], 2) - pow(k_left_arr[ix] + i, 2) * pow(dx, 2) * (dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * S_left[ix] + (dCpoDx_left[ix] + dCpoDx_right[ix]) * pow(k_left_arr[ix] + i, 3) * pow(dx, 3)) / pow(S_left[ix] - S_right[ix], 2);
            }
        }
    }
}

//Source_eq_y_3D << <1, N_cryst + 1 >> >  (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_bot, dCpoDx_top, S_top, S_bot, S_left, S_right, dCpoDx_right, dCpoDx_left, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);
                                        //(d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_bot, dCpoDx_top, S_top, S_bot, S_left, S_right, dCpoDx_right, dCpoDx_left, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);
                                        //(d_C, NX, NY, C_sat, dx, D_nd,dCpoDx_left, dCpoDx_right, S_right, S_left, S_bot, S_top, dCpoDx_top, dCpoDx_bot, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr)
__global__
void Source_eq_y_3D(double* d_C, int NX, int NY, double C_sat, double dx, double D_nd, double* dCpoDx_left, double* dCpoDx_right, double* S_right, double* S_left, double* S_bot, double* S_top, double* dCpoDx_top, double* dCpoDx_bot, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int* k_forth_arr, int* k_back_arr) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < k_right_arr[ix] - k_left_arr[ix]; i++) {
        for (int j = 0; j < k_top_arr[ix] - k_bot_arr[ix]; j++) {
            for (int k = 0; k < k_forth_arr[ix] - k_back_arr[ix]; k++) {
                d_C[NX * NY * (k_back_arr[ix] + k) + NX * (k_bot_arr[ix] + j) + k_left_arr[ix] + i] = ((-S_left[ix] * dCpoDx_left[ix] + (k_bot_arr[ix] + j) * dx * dCpoDx_left[ix] + C_sat) * pow(S_right[ix], 2) + (-pow(S_left[ix], 2) * dCpoDx_right[ix] + ((2 * dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * (k_bot_arr[ix] + j) * dx - 2 * C_sat) * S_left[ix] - (2. * (dCpoDx_left[ix] + dCpoDx_right[ix] / (2.))) * pow(k_bot_arr[ix] + j, 2) * pow(dx, 2)) * S_right[ix] + ((k_bot_arr[ix] + j) * dx * dCpoDx_right[ix] + C_sat) * pow(S_left[ix], 2) - pow(k_bot_arr[ix] + j, 2) * pow(dx, 2) * (dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * S_left[ix] + (dCpoDx_left[ix] + dCpoDx_right[ix]) * pow(k_bot_arr[ix] + j, 3) * pow(dx, 3)) / pow(S_left[ix] - S_right[ix], 2);
            }
        }
    }
}


__global__
void Source_eq_z_3D(double* d_C, int NX, int NY, double C_sat, double dx, double D_nd, double* dCpoDx_left, double* dCpoDx_right, double* S_right, double* S_left, double* S_bot, double* S_top, double* dCpoDx_top, double* dCpoDx_bot, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int* k_forth_arr, int* k_back_arr) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < k_right_arr[ix] - k_left_arr[ix]; i++) {
        for (int j = 0; j < k_top_arr[ix] - k_bot_arr[ix]; j++) {
            for (int k = 0; k < k_forth_arr[ix] - k_back_arr[ix]; k++) {
                d_C[NX * NY * (k_back_arr[ix] + k) + NX * (k_bot_arr[ix] + j) + k_left_arr[ix] + i] = ((-S_left[ix] * dCpoDx_left[ix] + (k_back_arr[ix] + k) * dx * dCpoDx_left[ix] + C_sat) * pow(S_right[ix], 2) + (-pow(S_left[ix], 2) * dCpoDx_right[ix] + ((2 * dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * (k_back_arr[ix] + k) * dx - 2 * C_sat) * S_left[ix] - (2. * (dCpoDx_left[ix] + dCpoDx_right[ix] / (2.))) * pow(k_back_arr[ix] + k, 2) * pow(dx, 2)) * S_right[ix] + ((k_back_arr[ix] + k) * dx * dCpoDx_right[ix] + C_sat) * pow(S_left[ix], 2) - pow(k_back_arr[ix] + k, 2) * pow(dx, 2) * (dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * S_left[ix] + (dCpoDx_left[ix] + dCpoDx_right[ix]) * pow(k_back_arr[ix] + k, 3) * pow(dx, 3)) / pow(S_left[ix] - S_right[ix], 2);
            }
        }
    }
}

__global__
void Coef_eq_g_x_3D(double* Rhs, int NX, int NY, double C_sat, double dx, double D_nd, double dCpoDx_left, double dCpoDx_right, double g, double S_right, int k_left, int k_bot, double S_left) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //Rhs[i + NY * j] = -g / D_nd * pow((k_left + i) * dx, 2.0) / 0.2e1 + 0.1e1 / D_nd * (abs(dCpoDx_right) * D_nd + abs(dCpoDx_left) * D_nd + 0.2e1 * g * S_right) * (k_left + i) * dx / 0.2e1 + (-D_nd * S_right * abs(dCpoDx_right) - D_nd * S_right * abs(dCpoDx_left) - g * pow(S_right, 2.0) + 0.2e1 * C_sat * D_nd) / D_nd / 0.2e1;
    //Rhs[i + NY * j] = (-((k_left + i) * dx - S_right) * ((k_left + i) * dx - S_left) * abs(dCpoDx_right) - ((k_left + i) * dx - S_right) * ((k_left + i) * dx - S_left) * abs(dCpoDx_left) + 2 * C_sat * (S_left - S_right)) / (2 * S_left - 2 * S_right);
    //Rhs[i + NY * j] = (-((k_left + i) * dx - S_right) * ((k_left + i) * dx - S_left) * abs(dCpoDx_right) - ((k_left + i) * dx - S_right) * ((k_left + i) * dx - S_left) * abs(dCpoDx_left) + 2 * C_sat * (S_left - S_right)) / (2 * S_left - 2 * S_right);
    Rhs[i + NY * j] = ((-S_left * dCpoDx_left + (k_left + i) * dx * dCpoDx_left + C_sat) * pow(S_right, 2) + (-pow(S_left, 2) * dCpoDx_right + ((2 * dCpoDx_left + 2 * dCpoDx_right) * (k_left + i) * dx - 2 * C_sat) * S_left - (2. * (dCpoDx_left + dCpoDx_right / (2.))) * pow(k_left + i, 2) * pow(dx, 2)) * S_right + ((k_left + i) * dx * dCpoDx_right + C_sat) * pow(S_left, 2) - pow(k_left + i, 2) * pow(dx, 2) * (dCpoDx_left + 2 * dCpoDx_right) * S_left + (dCpoDx_left + dCpoDx_right) * pow(k_left + i, 3) * pow(dx, 3)) / pow(S_left - S_right, 2);
}

__global__
void Coef_eq_g_y_3D(double* Rhs, int NX, int NY, double C_sat, double dx, double D_nd, double dCpoDx_left, double dCpoDx_right, double g, double S_right, int k_left, int k_bot, double S_left) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //Rhs[i + NY * j] = -g / D_nd * pow((k_bot + j) * dx, 2.0) / 0.2e1 + 0.1e1 / D_nd * (abs(dCpoDx_right) * D_nd + abs(dCpoDx_left) * D_nd + 0.2e1 * g * S_right) * (k_bot + j) * dx / 0.2e1 + (-D_nd * S_right * abs(dCpoDx_right) - D_nd * S_right * abs(dCpoDx_left) - g * pow(S_right, 2.0) + 0.2e1 * C_sat * D_nd) / D_nd / 0.2e1;
    //Rhs[i + NY * j] = (-((k_bot + j) * dx - S_right) * ((k_bot + j) * dx - S_left) * abs(dCpoDx_right) - ((k_bot + j) * dx - S_right) * ((k_bot + j) * dx - S_left) * abs(dCpoDx_left) + 2 * C_sat * (S_left - S_right)) / (2 * S_left - 2 * S_right);
    //Rhs[i + NY * j] = ((-S_left * dCpoDx_left + (k_bot + j) * dx * dCpoDx_left + C_sat) * pow(S_right, 2) + (-pow(S_left, 2) * dCpoDx_right + ((2 * dCpoDx_left + 2 * dCpoDx_right) * (k_bot + j) * dx - 2 * C_sat) * S_left - (2 * (dCpoDx_left + (1 / 2) * dCpoDx_right)) * pow((k_bot + j) * dx, 2)) * S_right + ((k_bot + j) * dx * dCpoDx_right + C_sat) * pow(S_left, 2) - pow((k_bot + j) * dx, 2) * (dCpoDx_left + 2 * dCpoDx_right) * S_left + (dCpoDx_left + dCpoDx_right) * pow((k_bot + j) * dx, 3)) / pow((S_left - S_right), 2.0);
    //Rhs[i + NY * j] = ((-S_left * dCpoDx_left + ((k_bot + j) * dx) * dCpoDx_left + C_sat) * pow( S_right , 2 )+ (-pow(S_left, 2) * dCpoDx_right + ((2 * dCpoDx_left + 2 * dCpoDx_right) * ((k_bot + j) * dx) - 2 * C_sat) * S_left - (2 * (dCpoDx_left + (1 / 2) * dCpoDx_right)) * pow(((k_bot + j) * dx), 2)) * S_right + (((k_bot + j) * dx) * dCpoDx_right + C_sat) * pow(S_left, 2) - pow(((k_bot + j) * dx), 2) * (dCpoDx_left + 2 * dCpoDx_right) * S_left + (dCpoDx_left + dCpoDx_right) * pow(((k_bot + j) * dx), 3)) / pow(S_left - S_right, 2);
    Rhs[i + NY * j] = ((-S_left * dCpoDx_left + (k_bot + j) * dx * dCpoDx_left + C_sat) * pow(S_right, 2) + (-pow(S_left, 2) * dCpoDx_right + ((2 * dCpoDx_left + 2 * dCpoDx_right) * (k_bot + j) * dx - 2 * C_sat) * S_left - (2. * (dCpoDx_left + dCpoDx_right / (2.))) * pow(k_bot + j, 2) * pow(dx, 2)) * S_right + ((k_bot + j) * dx * dCpoDx_right + C_sat) * pow(S_left, 2) - pow(k_bot + j, 2) * pow(dx, 2) * (dCpoDx_left + 2 * dCpoDx_right) * S_left + (dCpoDx_left + dCpoDx_right) * pow(k_bot + j, 3) * pow(dx, 3)) / pow(S_left - S_right, 2);
}

__global__
void distance_summer_3D(double* distance, int NX, int NY, int NZ, int N_cryst, double* S_cent_x, double* S_cent_y, double* S_cent_z, double gap, int k) {
    int i_k = blockIdx.x * blockDim.x + threadIdx.x;
    int j_k = blockIdx.y * blockDim.y + threadIdx.y;

    //distance[i_k + j_k * NX] = 0;
    for (int N_count = 0; N_count <= N_cryst; N_count++) {
        //distance[i_k + j_k * NX] = distance[i_k + j_k * NX] + fabs((i_k / (double)NX) - S_cent_x[N_count]) + fabs((j_k / (double)NY) - S_cent_y[N_count]);
        distance[i_k + j_k * NX] = fabs((i_k / (double)NX) - S_cent_x[N_count]) + fabs(j_k / (double)NY - S_cent_y[N_count]) + fabs(k / (double)NZ - S_cent_z[N_count]);
    }
}

__global__
void distance_summer_3D_optimized(double* distance, int NX, int NY, int NZ, int N_cryst, double* S_cent_x, double* S_cent_y, double* S_cent_z, double gap) {
    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    //distance[i_k + j_k * NX] = 0;
    for (int N_count = 0; N_count <= N_cryst; N_count++) {
        //distance[i_k + j_k * NX] = distance[i_k + j_k * NX] + fabs((i_k / (double)NX) - S_cent_x[N_count]) + fabs((j_k / (double)NY) - S_cent_y[N_count]);
        distance[i_k] = fabs((((i_k% NX))/ (double)NX) - S_cent_x[N_count]) + fabs(((i_k%(NX * NY))/NY) / (double)NY - S_cent_y[N_count]) + fabs((i_k / (NX * NY)) / (double)NZ - S_cent_z[N_count]);
    }
}

__global__
void distance_obnulator_3D(double* distance, double* d_C, int NX, int NY, int NZ, int N_cryst, double* S_cent_x, double* S_cent_y, double* S_cent_z, double gap, double max_val, double eps) {

    for (int i_k = gap; i_k <= NX - gap; i_k++) {
        for (int j_k = gap; j_k <= NY - gap; j_k++) {
            for (int k_k = gap; k_k <= NZ - gap; k_k++) {
                if (d_C[NX * NY * k_k + NX * j_k + i_k] < max_val - eps)
                    distance[NX * NY * k_k + NX * j_k + i_k] = 0;
            }
        }
    }
}

__global__
void distance_obnulator_3D_optimized(double* distance, double* d_C, int NX, int NY, int NZ, int N_cryst, double* S_cent_x, double* S_cent_y, double* S_cent_z, double gap, double max_val, double eps) {
    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    if (d_C[i_k] < max_val - eps) {
        distance[i_k] = 0;
    }
}

__global__
void Crystals_growth_3D(double* S_top, double* S_bot, double* S_left, double* S_right, double* S_back, double* S_forth, double* V_top, double* V_bot, double* V_left, double* V_right, double* V_back, double* V_forth, double dt) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    S_left[ix]  = S_left[ix]  + V_left[ix]  * dt;
    S_right[ix] = S_right[ix] + V_right[ix] * dt;
    S_top[ix]   = S_top[ix]   + V_top[ix]   * dt;
    S_bot[ix]   = S_bot[ix]   + V_bot[ix]   * dt;
    S_back[ix]  = S_back[ix]  + V_back[ix]  * dt;
    S_forth[ix] = S_forth[ix] + V_forth[ix] * dt;
}

__global__
void Coef_eq_new_cryst_3D(double* d_C, int NX, int NY, double init_value, int N_cryst, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int* k_back_arr, int* k_forth_arr){
    for (int i = k_left_arr[N_cryst]; i <  k_right_arr[N_cryst]; i++) {
        for (int j = k_bot_arr[N_cryst]; j < k_top_arr[N_cryst]; j++) {
            for (int z = k_back_arr[N_cryst]; z < k_forth_arr[N_cryst]; z++) {
                d_C[NX * NY * z + NX * j + i] = init_value;
            }
        }
    }
}

__global__
void Coef_a_zeros_x_3D(double* Ud, int NX, int NY, double init_value, int N_cryst, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int* k_back_arr, int* k_forth_arr, int* S_cent_x_num) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < k_top_arr[ix] - k_bot_arr[ix]; i++) {
        for (int k = 0; k < k_forth_arr[ix] - k_back_arr[ix]; k++) {
            //Ud[(NX + 2) * (NY + 2) * (k_back_arr[ix] + k ) + (S_cent_x_num[ix] + 0 + 1) + (k_bot_arr[ix] + 1 + i) * (NX + 2)] = init_value;
            Ud[(NX + 2) * (NY + 2) * (k_back_arr[ix] + k) + (S_cent_x_num[ix] + 0 + 1) * (NX + 2) + (k_bot_arr[ix] + 1 + i)] = init_value;
        }
    }
}

__global__
void Coef_a_zeros_y_3D(double* Ud, int NX, int NY, double init_value, int N_cryst, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int* k_back_arr, int* k_forth_arr, int* S_cent_y_num) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < k_right_arr[ix]-k_left_arr[ix]; i++) {
        for (int k = 0; k < k_forth_arr[ix] - k_back_arr[ix]; k++) {
            Ud[(NX + 2) * (NY + 2) * (k_back_arr[ix] +  k) + k_left_arr[ix] + (S_cent_y_num[ix] + 0 + 1) * (NX + 2) + 1 + i] = init_value;
        }
    }
}

__global__
void Coef_a_zeros_z_3D(double* Ud, int NX, int NY, double init_value, int N_cryst, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int* k_back_arr, int* k_forth_arr, int* S_cent_z_num) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < k_top_arr[ix] - k_bot_arr[ix]; i++) {
        for (int k = 0; k < k_right_arr[ix] - k_left_arr[ix]; k++) {
            Ud[(NX + 2) * (S_cent_z_num[ix] +1 ) + (k_bot_arr[ix] + 1 + i)  + (NX + 2) * (NY + 2) * (k_left_arr[ix] + 0  + k) ] = init_value;
        }
    }
}

//Coef_eq_2end << < dim3(k_top_arr[N_count] - k_bot_arr[N_count], 1), 1 >> > (&Ld[k_bot_arr[N_count] + (S_cent_x_num[N_count] - 1) * (NX + 2)], NX + 2, NY + 2, 0);


/******************************/
/*      Write in file       */
/******************************/
void WriteInFile_3D(double* h_T_GPU_result, int NX, int NY, int NZ, int name) {
    // --- Write in file
    char filename_format[] = "%d.dat";
    char filename[sizeof(filename_format) + 3];  // for up to 4 digit numbers
    snprintf(filename, sizeof(filename), filename_format, name);

    FILE* pointer = fopen(filename, "wb");
    // test for files not existing. 
    if (pointer == NULL || pointer == NULL)
    {
        printf("Error! Could not open file\n");
        exit(-1); // must include stdlib.h 
    }
    for (int i = 0; i < NY; i++) {
        for (int j = 0; j < NX; j++) {
            for (int k = 0; k < NZ; k++) {
                fprintf(pointer, "%.10f ", h_T_GPU_result[NX * NY * i + NX * j + k]);
            }
        }
        putc('\n', pointer);
    }
    fclose(pointer);
}

void WriteInFile_progress(int progress) {
    // --- Write in file
    char name[] = "progress.txt";
    //char filename[sizeof(filename_format) + 3];  // for up to 4 digit numbers
    //snprintf(filename, sizeof(filename), filename_format, name);

    FILE* pointer = fopen(name, "wb");
    // test for files not existing. 
    if (pointer == NULL || pointer == NULL)
    {
        printf("Error! Could not open file\n");
        exit(-1); // must include stdlib.h 
    }

    fprintf(pointer, "%d", progress);

    fclose(pointer);
}



__device__ void atomicMax_3D(double* const address, const double value)
{
    if (*address >= value)
    {
        return;
    }

    int* const addressAsI = (int*)address;
    int old = *addressAsI, assumed;

    do
    {
        assumed = old;
        if (__int2double_rn(assumed) >= value)
        {
            break;
        }

        old = atomicCAS(addressAsI, assumed, __double2int_rd(value));
    } while (assumed != old);
}


__global__ void reduceMaxIdxOptimized_3D(const double* __restrict__ input, const int size, double* maxOut, int* maxIdxOut)
{
    double localMax = 0.1;
    int localMaxIdx = 0;

    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        double val = input[i];

        if (localMax < abs(val))
        {
            localMax = abs(val);
            localMaxIdx = i;
        }
    }

    atomicMax(maxOut, localMax);

    __syncthreads();

    if (*maxOut == localMax)
    {
        *maxIdxOut = localMaxIdx;
    }
}


__global__ void reduceMaxIdx_3D(const double*  input, const int size, double* maxOut, int* maxIdxOut, int gap)
{
    double max = 0;
    int maxIdx = 0;

    for (int i = 0; i < size; i++)
    {
        if ((fabs(input[i]) > max) && (i % NX > gap ) && (i % NX < (NX - gap) ) && ( ((i % (NX * NY)) / NX) > gap) && (((i % (NX * NY)) / NX) < (NX - gap)) && ((i / (NX * NY)) > gap) && ((i / (NX * NY)) < (NX - gap)))
        {
            max = fabs(input[i]);
            maxIdx = i;
        }
    }

    maxOut[0] = max;
    maxIdxOut[0] = maxIdx;
}

__global__ void VecMax(const double* A, double* B, int N)
{
    extern __shared__ double cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    double temp = 0;  // register for each thread
    while (i < N) {
        if (A[i] > temp)
            temp = A[i];
        i += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;   // set the cache value 

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m

    int ib = blockDim.x / 2;
    while (ib != 0) {
        if (cacheIndex < ib && cache[cacheIndex + ib] > cache[cacheIndex])
            cache[cacheIndex] = cache[cacheIndex + ib];

        __syncthreads();

        ib /= 2;
    }

    if (cacheIndex == 0)
        B[blockIdx.x] = cache[0];
}


__global__
void Renewer_Ld_Dd_Ud_3D(double* Ld, double* Dd, double* Ud, int NX,int NY, double a, double b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Ld[i + 1 + NY * (j + 1)] = a;
    Dd[i + 1 + NY * (j + 1)] = b;
    Ud[i + 1 + NY * (j + 1)] = a;
}

__global__
void Renewer_Ld_Dd_Ud_Rhs_3D(double* Ld, double* Dd, double* Ud, int NX, int NY, double a, double b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Ld[i + 1 + NY * (j + 1)] = a;
    Dd[i + 1 + NY * (j + 1)] = b;
    Ud[i + 1 + NY * (j + 1)] = a;
}

//Coef_eq_arr_y << < dim3(k_right - k_left, 1), dim3(1, k_top - k_bot) >> > (&g[k_left + NX * k_bot], NX, NY, Coef_1, Coef_2, dy);
__global__
void Coef_eq_arr_g_x_3D(double* g, int NX, int NY, double  dy, int* k_left_arr, int* k_right_arr, int* k_bot_arr, int* k_top_arr, double* dCpoDx_bot, double* dCpoDx_top, double* dCpoDx_left, double* dCpoDx_right, double* S_top, double* S_bot, double* S_left, double* S_right, double D_nd) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    double g_avrg = D_nd * ((abs(dCpoDx_bot[ix]) + abs(dCpoDx_top[ix])) / (S_bot[ix] - S_top[ix]));

    double Coef_1 = -6 * D_nd * (dCpoDx_bot[ix] + dCpoDx_top[ix]) / (pow(S_bot[ix], 2) - 2 * S_bot[ix] * S_top[ix] + pow(S_top[ix], 2));
    double Coef_2 = (2 * (S_bot[ix] * dCpoDx_bot[ix] + 2 * S_bot[ix] * dCpoDx_top[ix] + 2 * S_top[ix] * dCpoDx_bot[ix] + S_top[ix] * dCpoDx_top[ix])) * D_nd / pow((S_bot[ix] - S_top[ix]), 2);

    
    for (int i = k_left_arr[ix]; i < k_right_arr[ix]; i++) {
        for (int j = k_bot_arr[ix]; j < k_top_arr[ix]; j++) {
            g[i + NY * j] = Coef_1 * j * dy + Coef_2;
        }
    }

}

__global__
void Coef_eq_arr_g_y_3D(double* g, int NX, int NY, double  dy, int* k_left_arr, int* k_right_arr, int* k_bot_arr, int* k_top_arr, double* dCpoDx_bot, double* dCpoDx_top, double* dCpoDx_left, double* dCpoDx_right, double* S_top, double* S_bot, double* S_left, double* S_right, double D_nd) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    double g_avrg = D_nd * (abs(dCpoDx_left[ix]) + abs(dCpoDx_right[ix]) / (S_left[ix] - S_right[ix]));


    double Coef_1 = -6 * D_nd * (dCpoDx_left[ix] + dCpoDx_right[ix]) / (pow(S_left[ix], 2) - 2 * S_left[ix] * S_right[ix] + pow(S_right[ix], 2));
    double Coef_2 = (2 * (S_left[ix] * dCpoDx_left[ix] + 2 * S_left[ix] * dCpoDx_right[ix] + 2 * S_right[ix] * dCpoDx_left[ix] + S_right[ix] * dCpoDx_right[ix])) * D_nd / pow((S_left[ix] - S_right[ix]), 2);

    for (int i = k_left_arr[ix]; i < k_right_arr[ix]; i++) {
        for (int j = k_bot_arr[ix]; j < k_top_arr[ix]; j++) {
            g[i + NY * j] = Coef_1 * i * dy + Coef_2;
        }
    }

}


//for (int k = 0; k <= NX; k++) {
//    buffer_d[k] = Coef_1 * k * dx + Coef_2;
//}

//-g / D_nd * x ^ 2 / 0.2e1 + 0.1e1 / D_nd * (abs((C_after_right(j) - C_befor_right(j)) / dx) * D_nd + abs((C_befor_left(j) - C_after_left(j)) / dx) * D_nd + 0.2e1 * g * S_right(j)) * x / 0.2e1 + (-D_nd * S_right(j) * abs((C_after_right(j) - C_befor_right(j)) / dx) - D_nd * S_right(j) * abs((C_befor_left(j) - C_after_left(j)) / dx) - g * S_right(j) ^ 2 + 0.2e1 * C0 * D_nd) / D_nd / 0.2e1;;
//Coef_eq << < 1, dim3(k_right - k_left, k_top - k_bot) >> > (&g[k_left + NX * k_bot], NX, NY, g_avrg);
// Coef_1 = 0;
// Coef_2 = C_sat;

#define MAX_CUDA_THREADS_PER_BLOCK 1024
#define NX_def NX


__global__ void Max_Interleaved_Addressing_Global(double* data, int data_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < data_size) {
        for (int stride = 1; stride < data_size; stride *= 2) {
            if (idx % (2 * stride) == 0) {
                double lhs = data[idx];
                double rhs = data[idx + stride];
                data[idx] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
}


__global__ void Max_Sequential_Addressing_Shared(double* data, int data_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size) {

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx];
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                double lhs = sdata[threadIdx.x];
                double rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) data[0] = sdata[0];
}

__global__ void Max_Sequential_Addressing_Shared_x(double* data, int data_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size) {

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx];    
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                double lhs = sdata[threadIdx.x];
                double rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) data[blockIdx.x] = sdata[0];
}

__global__ void Max_Sequential_Addressing_Shared_x_gap(double* data, int data_size, int gap) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size && (idx % NX > gap) && (idx % NX < (NX - gap)) && (((idx % (NX * NY)) / NX) > gap) && (((idx % (NX * NY)) / NX) < (NX - gap)) && ((idx / (NX * NY)) > gap) && ((idx / (NX * NY)) < (NX - gap))) {

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx];
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                double lhs = sdata[threadIdx.x];
                double rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) data[blockIdx.x] = sdata[0];
}

__global__ void Max_Sequential_Addressing_Shared_y(double* data, int data_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size) {

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx * 1024];
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                double lhs = sdata[threadIdx.x];
                double rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) data[0] = sdata[0];
}

__global__ void Max_Sequential_Addressing_Shared_z(double* data, int data_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size) {

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx * NX_def];
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                double lhs = sdata[threadIdx.x];
                double rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) data[0] = sdata[0];
}


/*
__global__ void Max_Sequential_Addressing_Shared(double* data, int data_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size) {


        sdata[threadIdx.x] = data[idx];
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                double lhs = sdata[threadIdx.x];
                double rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) data[0] = sdata[0];
}
*/

 
__global__ void Max_Sequential_Addressing_Shared_x_optimized(double* data_in, double* data_out,int data_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size) {

        /*copy to shared memory*/
        sdata[threadIdx.x] = data_in[idx];
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                double lhs = sdata[threadIdx.x];
                double rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) { 
        data_out[blockIdx.x] = sdata[0];
    };
}

__global__ void Max_Sequential_Addressing_Shared_x_optimized_gap(double* data_in, double* data_out, int data_size, int gap) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size && (idx % NX > gap) && (idx % NX < (NX - gap)) && (((idx % (NX * NY)) / NX) > gap) && (((idx % (NX * NY)) / NX) < (NX - gap)) && ((idx / (NX * NY)) > gap) && ((idx / (NX * NY)) < (NX - gap))) {

        /*copy to shared memory*/
        sdata[threadIdx.x] = data_in[idx];
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                double lhs = sdata[threadIdx.x];
                double rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        data_out[blockIdx.x] = sdata[0];
    };
}

__global__ void Max_Sequential_Addressing_Shared_x_optimized_index(double* data_in, double* data_out, int* data_index, int data_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double sdata[MAX_CUDA_THREADS_PER_BLOCK];
    __shared__ int sdata_index[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size) {

        /*copy to shared memory*/
        sdata[threadIdx.x] = data_in[idx];
        sdata[threadIdx.x] = idx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                double lhs = sdata[threadIdx.x];
                double rhs = sdata[threadIdx.x + stride];
                if (lhs < rhs) {
                    sdata[threadIdx.x] = rhs;

                }
                else {
                    sdata[threadIdx.x] = lhs;
                }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        data_out[blockIdx.x] = sdata[0];
    };
}

__global__ void Max_Sequential_Addressing_Shared_x_optimized_index_gap(double* data_in, double* data_out, int* data_index, int data_size, int gap) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double sdata[MAX_CUDA_THREADS_PER_BLOCK];
    __shared__ int sdata_index[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size) {

        /*copy to shared memory*/
        sdata[threadIdx.x] = data_in[idx];
        sdata[threadIdx.x] = idx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                double lhs = sdata[threadIdx.x];
                double rhs = sdata[threadIdx.x + stride];
                if (lhs < rhs) {
                    sdata[threadIdx.x] = rhs;

                }
                else {
                    sdata[threadIdx.x] = lhs;
                }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        data_out[blockIdx.x] = sdata[0];
    };
}