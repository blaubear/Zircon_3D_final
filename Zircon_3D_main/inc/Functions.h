inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}




__global__
void Rhs_GPU_diff_x(double* Rhs, double* C_old, double* divQ, double* g, int NX, int NY, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //Rhs[i + NY * j] = C_old[j + NY * i] - dt/2* divQ[j + NY * i];
    Rhs[i + 1 + NX * (j + 1)] = C_old[j + (NX - 2) * i] + 0* dt / 2 * g[j + (NX - 2) * i];
}

__global__
void Rhs_GPU_diff_y(double* Rhs, double* C_old, double* divQ, double* g, int NX, int NY, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //Rhs[i + NY * j] = C_old[i + NY * j] - dt / 2 * divQ[i + NY * j];
    Rhs[i + 1 + NX * (j + 1)] = C_old[i + (NX - 2) * j] + 0*dt / 2 * g[i + (NX - 2) * j];
}


__global__
void Coef_eq(double* Rhs, int NX, int NY, double eq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Rhs[i + NY * j] = eq;
}

__global__
void Coef_eq_i_1_j_1(double* Rhs, int NX, int NY, double eq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Rhs[i + 1 + NY * (j + 1)] = eq;
}

__global__
void Coef_eq_i_1_j_0(double* Rhs, int NX, int NY, double eq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Rhs[i + 1 + NY * (j)] = eq;
}

__global__
void Copy(double* T, double* x, int NX, int NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    T[i + NX * (j)] = x[i + NX * (j + 1)];
}

__global__
void Copy_rev(double* T, double* x, int NX, int NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    T[i + NX * (j)] = x[j + NX * (i + 1)];
}



/******************************/
/*      INITIALIZATION        */
/******************************/

void Initialize(double* h_T, const int NX, const int NY, double value)
{
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            //h_T[i * NY +j ] = i/128.0;
            h_T[i  + j * NY] = value;
            
        }
    }
}

__global__
void Coef_eq_addition(double* Rhs, int NX, int NY, double eq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Rhs[i + 1 + NY * (j + 1)] = Rhs[i + 1 + NY * (j + 1)] - eq;
}

__global__
void distance_counter(double* distance, int NX, int NY, double eq, int N_cryst, double* S_cent_x, double* S_cent_y) {
    int i_k = blockIdx.x * blockDim.x + threadIdx.x;
    int j_k = blockIdx.y * blockDim.y + threadIdx.y;
    distance[i_k + j_k * NX] = 0;
    //for (int N_count = 0; N_count <= N_cryst; N_count++) {
   //     distance[i_k + j_k * NX] = distance[i_k + j_k * NX] + pow(pow(fabs((i_k / (double)NX) - S_cent_x[N_count]), 1.0) + pow(fabs((j_k / (double)NY) - S_cent_y[N_count]), 1.0), 0.1);
    //}
}

__global__
void create_crystal(double S_x, double S_y, double* S_cent_x, double* S_cent_y, double* S_left, double* S_right, double* S_top, double* S_bot, double S_0, int N_cryst, int* S_cent_x_num, int* S_cent_y_num, double* Cryst_age, double time) {
    S_cent_x[N_cryst]     = S_x;
    S_cent_y[N_cryst]     = S_y;
    S_left[N_cryst]       = S_x - S_0;
    S_right[N_cryst]      = S_x + S_0;
    S_bot[N_cryst]        = S_y - S_0;
    S_top[N_cryst]        = S_y + S_0;
    S_cent_x_num[N_cryst] = (int)(S_x* (NX - 1));
    S_cent_y_num[N_cryst] = (int)(S_y *(NX - 1));
    Cryst_age[N_cryst]    = time;
}


double find_max(double* array, int NX, int NY, int gap, double* answer) {
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
void k_refresh(int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, double* S_left, double* S_right, double* S_top, double* S_bot, int NX, int NY, int* S_cent_x_num, int* S_cent_y_num, double dx) {
        int i_k = blockIdx.x * blockDim.x + threadIdx.x;

        k_left_arr[i_k] = S_cent_x_num[i_k];
        k_right_arr[i_k] = S_cent_x_num[i_k];
        k_top_arr[i_k] = S_cent_y_num[i_k];
        k_bot_arr[i_k] = S_cent_y_num[i_k];


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

}

__global__
void S_bot_refresh(double* d_C, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int NX, int NY, double D_nd, double dt, double dx, double* V_bot, double* dCpoDx_bot, double* summ, double C_sat, double C_cryst, double a, int FLAG_V_limit, int* FLAGGG_counter) {

    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    summ[0] = 0;
    for (int N_count_in = 0; N_count_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in++) {
        summ[0] = summ[0] + d_C[k_left_arr[i_k] + N_count_in + NX * (k_bot_arr[i_k] - 1)];
    }

    summ[1] = 0;
    for (int N_count_in = 0; N_count_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in++) {
        summ[1] = summ[1] + d_C[k_left_arr[i_k] + N_count_in + NX * (k_bot_arr[i_k])];
    }

    dCpoDx_bot[i_k] = -(summ[0] - summ[1]) / (k_right_arr[i_k] - k_left_arr[i_k]) / dx;

    V_bot[i_k] = -D_nd * dCpoDx_bot[i_k] / (C_sat - C_cryst);
    if (FLAG_V_limit){
        if (V_bot[i_k] < -a * pow((D_nd / dt ), (0.5)) || V_bot[i_k] > 0) {
            V_bot[i_k] = -a * pow((D_nd / dt ), (0.5));
            dCpoDx_bot[i_k] = -V_bot[i_k] * (C_sat - C_cryst) / D_nd;
            FLAGGG_counter[0] = FLAGGG_counter[0] + 1;
        }
    }
}


__global__
void S_top_refresh(double* d_C, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int NX, int NY, double D_nd, double dt, double dx, double* V_top, double* dCpoDx_top, double* summ, double C_sat, double C_cryst, double a, int FLAG_V_limit, int* FLAGGG_counter) {

    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    summ[0] = 0;
    for (int N_count_in = 0; N_count_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in++) {
        summ[0] = summ[0] + d_C[k_left_arr[i_k] + N_count_in + NX * (k_top_arr[i_k] - 1)];
    }

    summ[1] = 0;
    for (int N_count_in = 0; N_count_in < k_right_arr[i_k] - k_left_arr[i_k]; N_count_in++) {
        summ[1] = summ[1] + d_C[k_left_arr[i_k] + N_count_in + NX * (k_top_arr[i_k])];
    }

    dCpoDx_top[i_k] = -(summ[0] - summ[1]) / (k_right_arr[i_k] - k_left_arr[i_k]) / dx;

    V_top[i_k] = -D_nd * dCpoDx_top[i_k] / (C_sat - C_cryst);
    if (FLAG_V_limit){
        if (V_top[i_k] > a * pow((D_nd / dt ), (0.5)) || V_top[i_k] < 0) {
            V_top[i_k] = a * pow((D_nd / dt ), (0.5));
            dCpoDx_top[i_k] = -V_top[i_k] * (C_sat - C_cryst) / D_nd;
            FLAGGG_counter[0] = FLAGGG_counter[0] + 1;
        }
    }
}


__global__
void S_left_refresh(double* d_C, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int NX, int NY, double D_nd, double dt, double dx, double* V_left, double* dCpoDx_left, double* summ, double C_sat, double C_cryst, double a, int FLAG_V_limit, int* FLAGGG_counter) {

    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    summ[0] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        summ[0] = summ[0] + d_C[k_left_arr[i_k] - 1 + NX * (k_bot_arr[i_k] + N_count_in)];
    }

    summ[1] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        summ[1] = summ[1] + d_C[k_left_arr[i_k] + NX * (k_bot_arr[i_k] + N_count_in)];
    }

    dCpoDx_left[i_k] = -(summ[0] - summ[1]) / (k_top_arr[i_k] - k_bot_arr[i_k]) / dx;

    V_left[i_k] = -D_nd * dCpoDx_left[i_k] / (C_sat - C_cryst);
    if (FLAG_V_limit){
        if (V_left[i_k] < -a * pow((D_nd / dt ), (0.5)) || V_left[i_k] > 0) {
            V_left[i_k] = -a * pow((D_nd / dt ), (0.5));
            dCpoDx_left[i_k] = -V_left[i_k] * (C_sat - C_cryst) / D_nd;
            FLAGGG_counter[0] = FLAGGG_counter[0] + 1;
        }
    }
}


__global__
void S_right_refresh(double* d_C, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int NX, int NY, double D_nd, double dt, double dx, double* V_right, double* dCpoDx_right, double* summ, double C_sat, double C_cryst, double a, int FLAG_V_limit, int* FLAGGG_counter) {

    int i_k = blockIdx.x * blockDim.x + threadIdx.x;

    summ[0] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        summ[0] = summ[0] + d_C[k_right_arr[i_k] - 1 + NX * (k_bot_arr[i_k] + N_count_in)];
    }

    summ[1] = 0;
    for (int N_count_in = 0; N_count_in < k_top_arr[i_k] - k_bot_arr[i_k]; N_count_in++) {
        summ[1] = summ[1] + d_C[k_right_arr[i_k] + NX * (k_bot_arr[i_k] + N_count_in)];
    }

    dCpoDx_right[i_k] = -(summ[0] - summ[1]) / (k_top_arr[i_k] - k_bot_arr[i_k]) / dx;

    V_right[i_k] = -D_nd * dCpoDx_right[i_k] / (C_sat - C_cryst);

    if (FLAG_V_limit){
        if (V_right[i_k] > a * pow((D_nd / dt ), (0.5)) || V_right[i_k] < 0) {
            V_right[i_k] = a * pow((D_nd / dt ), (0.5));
            dCpoDx_right[i_k] = -V_right[i_k] * (C_sat - C_cryst) / D_nd;
            FLAGGG_counter[0] = FLAGGG_counter[0] + 1;
        }
    } 
}

__global__
void Source_eq_y(double* d_C, int NX, int NY, double C_sat, double h_nd, double D_nd, double* dCpoDx_left, double* dCpoDx_right, double* S_right, double* S_left, double* S_bot, double* S_top, double* dCpoDx_top, double* dCpoDx_bot, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < k_right_arr[ix] - k_left_arr[ix]; i++) {
        for (int j = 0 ; j < k_top_arr[ix] - k_bot_arr[ix]; j++) {
            //d_C[k_left_arr[ix] + i + NX * (k_bot_arr[ix] + j)] = ((-S_left[ix] * dCpoDx_left[ix] + (k_bot_arr[ix] + j) * h_nd * dCpoDx_left[ix] + C_sat) * pow(S_right[ix], 2) + (-pow(S_left[ix], 2) * dCpoDx_right[ix] + ((2 * dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * (k_bot_arr[ix] + j) * h_nd - 2 * C_sat) * S_left[ix] - (2. * (dCpoDx_left[ix] + dCpoDx_right[ix] / (2.))) * pow(k_bot_arr[ix] + j, 2) * pow(h_nd, 2)) * S_right[ix] + ((k_bot_arr[ix] + j) * h_nd * dCpoDx_right[ix] + C_sat) * pow(S_left[ix], 2) - pow(k_bot_arr[ix] + j, 2) * pow(h_nd, 2) * (dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * S_left[ix] + (k_[ix] + dCpoDx_right[ix]) * pow(k_bot_arr[ix] + j, 3) * pow(h_nd, 3)) / pow(S_left[ix] - S_right[ix], 2);
            d_C[k_left_arr[ix] + i + NX * (k_bot_arr[ix] + j)] = ((-S_left[ix] * dCpoDx_left[ix] + (k_bot_arr[ix] + j) * h_nd * dCpoDx_left[ix] + C_sat) * pow(S_right[ix], 2) + (-pow(S_left[ix], 2) * dCpoDx_right[ix] + ((2 * dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * (k_bot_arr[ix] + j) * h_nd - 2 * C_sat) * S_left[ix] - (2. * (dCpoDx_left[ix] + dCpoDx_right[ix] / (2.))) * pow(k_bot_arr[ix] + j, 2) * pow(h_nd, 2)) * S_right[ix] + ((k_bot_arr[ix] + j) * h_nd * dCpoDx_right[ix] + C_sat) * pow(S_left[ix], 2) - pow(k_bot_arr[ix] + j, 2) * pow(h_nd, 2) * (dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * S_left[ix] + (dCpoDx_left[ix] + dCpoDx_right[ix]) * pow(k_bot_arr[ix] + j, 3) * pow(h_nd, 3)) / pow(S_left[ix] - S_right[ix], 2);
        }
    }
}

__global__
void Source_eq_x(double* d_C, int NX, int NY, double C_sat, double h_nd, double D_nd, double* dCpoDx_left, double* dCpoDx_right, double* S_right, double* S_left, double* S_bot, double* S_top, double* dCpoDx_top, double* dCpoDx_bot, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < k_right_arr[ix] - k_left_arr[ix]; i++) {
        for (int j = 0; j < k_top_arr[ix] - k_bot_arr[ix]; j++) {
            d_C[k_left_arr[ix] + i + NX * (k_bot_arr[ix] + j)] = ((-S_left[ix] * dCpoDx_left[ix] + (k_left_arr[ix] + i) * h_nd * dCpoDx_left[ix] + C_sat) * pow(S_right[ix], 2) + (-pow(S_left[ix], 2) * dCpoDx_right[ix] + ((2 * dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * (k_left_arr[ix] + i) * h_nd - 2 * C_sat) * S_left[ix] - (2. * (dCpoDx_left[ix] + dCpoDx_right[ix] / (2.))) * pow(k_left_arr[ix] + i, 2) * pow(h_nd, 2)) * S_right[ix] + ((k_left_arr[ix] + i) * h_nd * dCpoDx_right[ix] + C_sat) * pow(S_left[ix], 2) - pow(k_left_arr[ix] + i, 2) * pow(h_nd, 2) * (dCpoDx_left[ix] + 2 * dCpoDx_right[ix]) * S_left[ix] + (dCpoDx_left[ix] + dCpoDx_right[ix]) * pow(k_left_arr[ix] + i, 3) * pow(h_nd, 3)) / pow(S_left[ix] - S_right[ix], 2);
        }
    }
}

__global__
void Coef_eq_g_x(double* Rhs, int NX, int NY, double C_sat, double h_nd, double D_nd, double dCpoDx_left, double dCpoDx_right, double g, double S_right, int k_left, int k_bot, double S_left) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //Rhs[i + NY * j] = -g / D_nd * pow((k_left + i) * h_nd, 2.0) / 0.2e1 + 0.1e1 / D_nd * (abs(dCpoDx_right) * D_nd + abs(dCpoDx_left) * D_nd + 0.2e1 * g * S_right) * (k_left + i) * h_nd / 0.2e1 + (-D_nd * S_right * abs(dCpoDx_right) - D_nd * S_right * abs(dCpoDx_left) - g * pow(S_right, 2.0) + 0.2e1 * C_sat * D_nd) / D_nd / 0.2e1;
    //Rhs[i + NY * j] = (-((k_left + i) * h_nd - S_right) * ((k_left + i) * h_nd - S_left) * abs(dCpoDx_right) - ((k_left + i) * h_nd - S_right) * ((k_left + i) * h_nd - S_left) * abs(dCpoDx_left) + 2 * C_sat * (S_left - S_right)) / (2 * S_left - 2 * S_right);
    //Rhs[i + NY * j] = (-((k_left + i) * h_nd - S_right) * ((k_left + i) * h_nd - S_left) * abs(dCpoDx_right) - ((k_left + i) * h_nd - S_right) * ((k_left + i) * h_nd - S_left) * abs(dCpoDx_left) + 2 * C_sat * (S_left - S_right)) / (2 * S_left - 2 * S_right);
    Rhs[i + NY * j] = ((-S_left * dCpoDx_left + (k_left + i) * h_nd * dCpoDx_left + C_sat) * pow(S_right, 2) + (-pow(S_left, 2) * dCpoDx_right + ((2 * dCpoDx_left + 2 * dCpoDx_right) * (k_left + i) * h_nd - 2 * C_sat) * S_left - (2. * (dCpoDx_left + dCpoDx_right / (2.))) * pow(k_left + i, 2) * pow(h_nd, 2)) * S_right + ((k_left + i) * h_nd * dCpoDx_right + C_sat) * pow(S_left, 2) - pow(k_left + i, 2) * pow(h_nd, 2) * (dCpoDx_left + 2 * dCpoDx_right) * S_left + (dCpoDx_left + dCpoDx_right) * pow(k_left + i, 3) * pow(h_nd, 3)) / pow(S_left - S_right, 2);
}

__global__
void Coef_eq_g_y(double* Rhs, int NX, int NY, double C_sat, double h_nd, double D_nd, double dCpoDx_left, double dCpoDx_right, double g, double S_right, int k_left, int k_bot, double S_left) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //Rhs[i + NY * j] = -g / D_nd * pow((k_bot + j) * h_nd, 2.0) / 0.2e1 + 0.1e1 / D_nd * (abs(dCpoDx_right) * D_nd + abs(dCpoDx_left) * D_nd + 0.2e1 * g * S_right) * (k_bot + j) * h_nd / 0.2e1 + (-D_nd * S_right * abs(dCpoDx_right) - D_nd * S_right * abs(dCpoDx_left) - g * pow(S_right, 2.0) + 0.2e1 * C_sat * D_nd) / D_nd / 0.2e1;
    //Rhs[i + NY * j] = (-((k_bot + j) * h_nd - S_right) * ((k_bot + j) * h_nd - S_left) * abs(dCpoDx_right) - ((k_bot + j) * h_nd - S_right) * ((k_bot + j) * h_nd - S_left) * abs(dCpoDx_left) + 2 * C_sat * (S_left - S_right)) / (2 * S_left - 2 * S_right);
    //Rhs[i + NY * j] = ((-S_left * dCpoDx_left + (k_bot + j) * h_nd * dCpoDx_left + C_sat) * pow(S_right, 2) + (-pow(S_left, 2) * dCpoDx_right + ((2 * dCpoDx_left + 2 * dCpoDx_right) * (k_bot + j) * h_nd - 2 * C_sat) * S_left - (2 * (dCpoDx_left + (1 / 2) * dCpoDx_right)) * pow((k_bot + j) * h_nd, 2)) * S_right + ((k_bot + j) * h_nd * dCpoDx_right + C_sat) * pow(S_left, 2) - pow((k_bot + j) * h_nd, 2) * (dCpoDx_left + 2 * dCpoDx_right) * S_left + (dCpoDx_left + dCpoDx_right) * pow((k_bot + j) * h_nd, 3)) / pow((S_left - S_right), 2.0);
    //Rhs[i + NY * j] = ((-S_left * dCpoDx_left + ((k_bot + j) * h_nd) * dCpoDx_left + C_sat) * pow( S_right , 2 )+ (-pow(S_left, 2) * dCpoDx_right + ((2 * dCpoDx_left + 2 * dCpoDx_right) * ((k_bot + j) * h_nd) - 2 * C_sat) * S_left - (2 * (dCpoDx_left + (1 / 2) * dCpoDx_right)) * pow(((k_bot + j) * h_nd), 2)) * S_right + (((k_bot + j) * h_nd) * dCpoDx_right + C_sat) * pow(S_left, 2) - pow(((k_bot + j) * h_nd), 2) * (dCpoDx_left + 2 * dCpoDx_right) * S_left + (dCpoDx_left + dCpoDx_right) * pow(((k_bot + j) * h_nd), 3)) / pow(S_left - S_right, 2);
    Rhs[i + NY * j] = ((-S_left * dCpoDx_left + (k_bot + j) * h_nd * dCpoDx_left + C_sat) * pow(S_right, 2) + (-pow(S_left, 2) * dCpoDx_right + ((2 * dCpoDx_left + 2 * dCpoDx_right) * (k_bot + j) * h_nd - 2 * C_sat) * S_left - (2. * (dCpoDx_left + dCpoDx_right / (2.))) * pow(k_bot + j, 2) * pow(h_nd, 2)) * S_right + ((k_bot + j) * h_nd * dCpoDx_right + C_sat) * pow(S_left, 2) - pow(k_bot + j, 2) * pow(h_nd, 2) * (dCpoDx_left + 2 * dCpoDx_right) * S_left + (dCpoDx_left + dCpoDx_right) * pow(k_bot + j, 3) * pow(h_nd, 3)) / pow(S_left - S_right, 2);
}

__global__
void distance_summer(double* distance, int NX, int NY, int N_cryst, double* S_cent_x, double* S_cent_y, double gap) {
    int i_k = blockIdx.x * blockDim.x + threadIdx.x;
    int j_k = blockIdx.y * blockDim.y + threadIdx.y;

    distance[i_k + j_k * NX] = 0;
    for (int N_count = 0; N_count <= N_cryst; N_count++) {
        //distance[i_k + j_k * NX] = distance[i_k + j_k * NX] + fabs((i_k / (double)NX) - S_cent_x[N_count]) + fabs((j_k / (double)NY) - S_cent_y[N_count]);
        distance[i_k + j_k * NX] = distance[i_k + j_k * NX] + fabs((i_k / (double)NX) - S_cent_x[N_count]) + fabs(j_k / (double)NY - S_cent_y[N_count]);
    }
}

__global__
void distance_obnulator(double* distance, double* d_C, int NX, int NY, int N_cryst, double* S_cent_x, double* S_cent_y, double gap, double max_val, double eps) {

    for (int i_k = gap; i_k <= NX - gap; i_k++) {
        for (int j_k = gap; j_k <= NX - gap; j_k++) {
            if (d_C[i_k + j_k * NX] < max_val - eps)
                distance[i_k + j_k * NX] = 0;
        }
    }

}

__global__
void Crystals_growth(double* S_top, double* S_bot, double* S_left, double* S_right, double* V_top, double* V_bot, double* V_left, double* V_right,double dt) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    S_left[ix]  = S_left[ix]  + V_left[ix]  * dt;
    S_right[ix] = S_right[ix] + V_right[ix] * dt;
    S_top[ix]   = S_top[ix]   + V_top[ix]   * dt;
    S_bot[ix]   = S_bot[ix]   + V_bot[ix]   * dt;
}

__global__
void Coef_eq_new_cryst(double* d_C, int NX, int NY, double init_value, int N_cryst, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr){
    for (int i = k_left_arr[N_cryst]; i <  k_right_arr[N_cryst]; i++) {
        for (int j = k_bot_arr[N_cryst]; j < k_top_arr[N_cryst]; j++) {
            d_C[i + NX * j] = init_value;
        }
    }
}

__global__
void Coef_a_zeros_y(double* Ud, int NX, int NY, double init_value, int N_cryst, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int* S_cent_y_num) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < k_right_arr[N_cryst]-k_left_arr[N_cryst]; i++) {
            Ud[k_left_arr[ix] + (S_cent_y_num[ix] + 0 + 1) * (NX + 2) + 1 + i] = init_value;
    }
}

__global__
void Coef_a_zeros_x(double* Ud, int NX, int NY, double init_value, int N_cryst, int* k_left_arr, int* k_right_arr, int* k_top_arr, int* k_bot_arr, int* S_cent_x_num) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < k_top_arr[N_cryst] - k_bot_arr[N_cryst]; i++) {
        Ud[k_bot_arr[ix] + (S_cent_x_num[ix] + 0 + 1) * (NX + 2) + 1 + i] = init_value;
    }
}
//Coef_eq_2end << < dim3(k_top_arr[N_count] - k_bot_arr[N_count], 1), 1 >> > (&Ld[k_bot_arr[N_count] + (S_cent_x_num[N_count] - 1) * (NX + 2)], NX + 2, NY + 2, 0);


/******************************/
/*      Write in file       */
/******************************/
void WriteInFile(double* h_T_GPU_result, int NX, int NY, int name) {
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
            fprintf(pointer, "%.10f ", h_T_GPU_result[NX * i + j]);
        }
        putc('\n', pointer);
    }
    fclose(pointer);
}

void WriteInFile_2D(double* h_T_GPU_result, int NX, int NY, int name) {
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
            fprintf(pointer, "%.10f ", h_T_GPU_result[NX * i + j]);
        }
        putc('\n', pointer);
    }
    fclose(pointer);
}





__device__ void atomicMax(double* const address, const double value)
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


__global__ void reduceMaxIdxOptimized(const double* __restrict__ input, const int size, double* maxOut, int* maxIdxOut)
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


__global__ void reduceMaxIdx(const double*  input, const int size, double* maxOut, int* maxIdxOut, int gap)
{
    double max = 0.0;
    int maxIdx = 0;

    for (int i = 0; i < size; i++)
    {
        if ((fabs(input[i]) > max) && (i % NX > gap ) && (i % NX < NX - gap ) && (i / NX > gap) && (i / NX < NX - gap))
        {
            max = fabs(input[i]);
            maxIdx = i;
        }
    }

    maxOut[0] = max;
    maxIdxOut[0] = maxIdx;
}




__global__
void Renewer_Ld_Dd_Ud(double* Ld, double* Dd, double* Ud, int NX,int NY, double a, double b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Ld[i + 1 + NY * (j + 1)] = a;
    Dd[i + 1 + NY * (j + 1)] = b;
    Ud[i + 1 + NY * (j + 1)] = a;
}

__global__
void Renewer_Ld_Dd_Ud_Rhs(double* Ld, double* Dd, double* Ud, int NX, int NY, double a, double b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Ld[i + 1 + NY * (j + 1)] = a;
    Dd[i + 1 + NY * (j + 1)] = b;
    Ud[i + 1 + NY * (j + 1)] = a;
}

//Coef_eq_arr_y << < dim3(k_right - k_left, 1), dim3(1, k_top - k_bot) >> > (&g[k_left + NX * k_bot], NX, NY, Coef_1, Coef_2, dy);
__global__
void Coef_eq_arr_g_x(double* g, int NX, int NY, double  dy, int* k_left_arr, int* k_right_arr, int* k_bot_arr, int* k_top_arr, double* dCpoDx_bot, double* dCpoDx_top, double* dCpoDx_left, double* dCpoDx_right, double* S_top, double* S_bot, double* S_left, double* S_right, double D_nd) {
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
void Coef_eq_arr_g_y(double* g, int NX, int NY, double  dy, int* k_left_arr, int* k_right_arr, int* k_bot_arr, int* k_top_arr, double* dCpoDx_bot, double* dCpoDx_top, double* dCpoDx_left, double* dCpoDx_right, double* S_top, double* S_bot, double* S_left, double* S_right, double D_nd) {
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


//-g / D_nd * x ^ 2 / 0.2e1 + 0.1e1 / D_nd * (abs((C_after_right(j) - C_befor_right(j)) / h_nd) * D_nd + abs((C_befor_left(j) - C_after_left(j)) / h_nd) * D_nd + 0.2e1 * g * S_right(j)) * x / 0.2e1 + (-D_nd * S_right(j) * abs((C_after_right(j) - C_befor_right(j)) / h_nd) - D_nd * S_right(j) * abs((C_befor_left(j) - C_after_left(j)) / h_nd) - g * S_right(j) ^ 2 + 0.2e1 * C0 * D_nd) / D_nd / 0.2e1;;
//Coef_eq << < 1, dim3(k_right - k_left, k_top - k_bot) >> > (&g[k_left + NX * k_bot], NX, NY, g_avrg);
// Coef_1 = 0;
// Coef_2 = C_sat;
