#include "stdio.h"
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Thomas_f.h"
#include "arrays.h"
#include "Functions.h"
#include "Functions_3D.h"
#include <assert.h>
#include <sys/types.h>
#include <stdlib.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


/********/
/* MAIN */
/********/
int main() {

    //инициализация
#ifdef INIT_VALUES


    int N_cryst = 0;
    int graph_number = 10;
    //double g_avrg = 0;
    double dt = 1 / (double)(Nt - 1);
    double dt_init = dt;
    double T = 0, D_nd = 0;
    double time = 0;
    int Counter = 0;
    int time_progress;
    //double Ly = Lx;
    cudaEvent_t start = 0, stop = 0;
    float elapsedTime = 0;
    printf("D_nd = %d \n", N_cryst);
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    //double dCpoDx = 0;
    //double dCpoDx_top = 0, dCpoDx_bot = 0, dCpoDx_left = 0, dCpoDx_right = 0;
    double max_val = 0;

    double* h_C_GPU_result = (double*)calloc((NX + 2) * (NY + 2) * (NZ + 2), sizeof(double));

    int max_val_x = 0, max_val_y = 0, max_val_z = 0;
    double max_val_x_s = 0, max_val_y_s = 0, max_val_z_s = 0;
    int i = 0;
    int dt_counter = 0;

    double* h_C = (double*)calloc(NX * NY * NZ, sizeof(double));        assert(h_C != NULL);
    double* h_C_old = (double*)calloc(NX * NY * NZ, sizeof(double));    assert(h_C_old != NULL);
    double* max_val_arr = (double*)calloc(1, sizeof(double));           assert(max_val_arr != NULL);

    Initialize_3D(h_C, NX, NY, NZ, init_value);

    //Initialize_3D_test(h_C, NX, NY, NZ, init_value);

    double pp = 0, p = 0;
    int p_int = 0;;

    double* arr_dt = (double*)malloc(N_dt_razb * sizeof(double));       assert(arr_dt != NULL);

    if (FLAG_S_0_increaser && (S_0 < dx)) {
        S_0 = dx;
    }

    for (int j = 1; j <= N_dt_razb; j++) {
        arr_dt[j - 1] = 1 * (1.0 / (double)N_dt_razb) * dt;
    }


    // --- GPU  distribution
    double* d_C;     cudaMalloc((void**)&d_C, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));       assert(d_C != NULL);
    double* d_C_old; cudaMalloc((void**)&d_C_old, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));   assert(d_C_old != NULL);

    // double* d_C_test; cudaMalloc((void**)&d_C_test, NX * NY * NZ * sizeof(double));   assert(d_C_test != NULL);

    double* d_max;      cudaMalloc((void**)&d_max, 2 * sizeof(double));             assert(d_max != NULL);
    int* d_max_index;      cudaMalloc((void**)&d_max_index, 2 * sizeof(int));       assert(d_max_index != NULL);




    //double* g;                cudaMalloc((void**)&g, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));    assert(g != NULL);
    double* g;                cudaMalloc((void**)&g, 1 * sizeof(double));    assert(g != NULL);
    double* divQ;             cudaMalloc((void**)&divQ, 1 * sizeof(double));          assert(divQ != NULL);

    double* qx;             cudaMalloc((void**)&qx, (NX) * (NY) * (NZ) * sizeof(double));          assert(qx != NULL);
    double* qy;             cudaMalloc((void**)&qy, (NX) * (NY) * (NZ) * sizeof(double));
    double* qz;             cudaMalloc((void**)&qz, (NX) * (NY) * (NZ) * sizeof(double));

    double* Qx;             cudaMalloc((void**)&Qx, (NX) * (NY) * (NZ) * sizeof(double));
    double* Qy;             cudaMalloc((void**)&Qy, (NX) * (NY) * (NZ) * sizeof(double));
    double* Qz;             cudaMalloc((void**)&Qz, (NX) * (NY) * (NZ) * sizeof(double));

    double* Ld;               cudaMalloc((void**)&Ld, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));     assert(Ld != NULL);
    double* Dd;               cudaMalloc((void**)&Dd, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));     assert(Dd != NULL);
    double* Ud;               cudaMalloc((void**)&Ud, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));     assert(Ud != NULL);
    double* Rhs;              cudaMalloc((void**)&Rhs, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));    assert(Rhs != NULL);
    double* x;                cudaMalloc((void**)&x, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));      assert(x != NULL);       //    double* x;                cudaMalloc((void**)&x,   (NX) * (NY + 2) * (NZ + 2) * sizeof(double));

    double* data_out;         cudaMalloc((void**)&data_out, (size / 1024) * sizeof(double));                        assert(data_out != NULL);

    int* FLAGGG_counter;      cudaMalloc((void**)&FLAGGG_counter, 1 * sizeof(int));

    //cudaMemcpy(g, h_C_GPU_result, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_C, h_C_GPU_result, (NX) * (NY) * (NZ) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_old, h_C_GPU_result, (NX) * (NY) * (NZ) * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_C_test, h_C_GPU_result, (NX) * (NY) * (NZ) * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(Ld, h_C_GPU_result, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Dd, h_C_GPU_result, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Ud, h_C_GPU_result, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Rhs, h_C_GPU_result, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x, h_C_GPU_result, (NX) * (NY + 2) * (NZ + 2) * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(data_out, h_C_GPU_result, (size / 1024) * sizeof(double), cudaMemcpyHostToDevice);

    int* k_left_arr;        cudaMalloc((void**)&k_left_arr, N_cryst_max * sizeof(int));         assert(k_left_arr != NULL);
    int* k_right_arr;       cudaMalloc((void**)&k_right_arr, N_cryst_max * sizeof(int));        assert(k_right_arr != NULL);
    int* k_top_arr;         cudaMalloc((void**)&k_top_arr, N_cryst_max * sizeof(int));          assert(k_top_arr != NULL);
    int* k_bot_arr;         cudaMalloc((void**)&k_bot_arr, N_cryst_max * sizeof(int));          assert(k_bot_arr != NULL);
    int* k_back_arr;         cudaMalloc((void**)&k_back_arr, N_cryst_max * sizeof(int));        assert(k_back_arr != NULL);
    int* k_forth_arr;         cudaMalloc((void**)&k_forth_arr, N_cryst_max * sizeof(int));      assert(k_forth_arr != NULL);

    double* arr_time_d = (double*)malloc(5 * (Nt + N_cryst_max * N_dt_razb) * sizeof(double));      assert(arr_time_d != NULL);
    double* arr_V_nucl_d = (double*)malloc(5 * (Nt + N_cryst_max * N_dt_razb) * sizeof(double));    assert(arr_V_nucl_d != NULL);
    double* arr_T_d = (double*)malloc(5 * (Nt + N_cryst_max * N_dt_razb) * sizeof(double));         assert(arr_T_d != NULL);


    double* S_cent_x;     cudaMalloc((void**)&S_cent_x, N_cryst_max * sizeof(double));          assert(S_cent_x != NULL);
    double* S_cent_y;     cudaMalloc((void**)&S_cent_y, N_cryst_max * sizeof(double));          assert(S_cent_y != NULL);
    double* S_cent_z;     cudaMalloc((void**)&S_cent_z, N_cryst_max * sizeof(double));          assert(S_cent_z != NULL);


    double* S_left;        cudaMalloc((void**)&S_left, N_cryst_max * sizeof(double));           assert(S_left != NULL);
    double* S_right;       cudaMalloc((void**)&S_right, N_cryst_max * sizeof(double));          assert(S_right != NULL);
    double* S_top;         cudaMalloc((void**)&S_top, N_cryst_max * sizeof(double));            assert(S_top != NULL);
    double* S_bot;         cudaMalloc((void**)&S_bot, N_cryst_max * sizeof(double));            assert(S_bot != NULL);
    double* S_back;        cudaMalloc((void**)&S_back, N_cryst_max * sizeof(double));           assert(S_back != NULL);
    double* S_forth;       cudaMalloc((void**)&S_forth, N_cryst_max * sizeof(double));          assert(S_forth != NULL);

    double* Cryst_age;     cudaMalloc((void**)&Cryst_age, N_cryst_max * sizeof(double));        assert(Cryst_age != NULL);

    int* S_cent_x_num;     cudaMalloc((void**)&S_cent_x_num, N_cryst_max * sizeof(int));        assert(S_cent_x_num != NULL);
    int* S_cent_y_num;     cudaMalloc((void**)&S_cent_y_num, N_cryst_max * sizeof(int));        assert(S_cent_y_num != NULL);
    int* S_cent_z_num;     cudaMalloc((void**)&S_cent_z_num, N_cryst_max * sizeof(int));        assert(S_cent_z_num != NULL);

    double* distance;     cudaMalloc((void**)&distance, NX * NY * NZ * sizeof(double));         assert(distance != NULL);

    double* V_left;        cudaMalloc((void**)&V_left, N_cryst_max * sizeof(double));           assert(V_left != NULL);
    double* V_right;       cudaMalloc((void**)&V_right, N_cryst_max * sizeof(double));          assert(V_right != NULL);
    double* V_top;         cudaMalloc((void**)&V_top, N_cryst_max * sizeof(double));            assert(V_top != NULL);
    double* V_bot;         cudaMalloc((void**)&V_bot, N_cryst_max * sizeof(double));            assert(V_bot != NULL);
    double* V_forth;         cudaMalloc((void**)&V_forth, N_cryst_max * sizeof(double));        assert(V_forth != NULL);
    double* V_back;         cudaMalloc((void**)&V_back, N_cryst_max * sizeof(double));          assert(V_back != NULL);

    double* summ;         cudaMalloc((void**)&summ, 2 * sizeof(double));                        assert(summ != NULL);

    double* buffer = (double*)malloc(NX * NY * NZ * sizeof(double));                            assert(buffer != NULL);

    double* d_summ;           cudaMalloc((void**)&d_summ, 2 * sizeof(double));                  assert(d_summ != NULL);


    double* dCpoDx_left;        cudaMalloc((void**)&dCpoDx_left, N_cryst_max * sizeof(double));     assert(dCpoDx_left != NULL);
    double* dCpoDx_right;       cudaMalloc((void**)&dCpoDx_right, N_cryst_max * sizeof(double));    assert(dCpoDx_right != NULL);
    double* dCpoDx_top;         cudaMalloc((void**)&dCpoDx_top, N_cryst_max * sizeof(double));      assert(dCpoDx_top != NULL);
    double* dCpoDx_bot;         cudaMalloc((void**)&dCpoDx_bot, N_cryst_max * sizeof(double));      assert(dCpoDx_bot != NULL);
    double* dCpoDx_forth;       cudaMalloc((void**)&dCpoDx_forth, N_cryst_max * sizeof(double));    assert(dCpoDx_forth != NULL);
    double* dCpoDx_back;        cudaMalloc((void**)&dCpoDx_back, N_cryst_max * sizeof(double));     assert(dCpoDx_back != NULL);



    cudaMemcpy(d_C, h_C, NX * NY * NZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_old, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToDevice);
    //cudaMemcpy(Rhs, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToDevice);
    //Initialize_3D(h_C, NX, NY, NZ, 0);
    //cudaMemcpy(distance, h_C, NX * NY * NZ * sizeof(double), cudaMemcpyHostToDevice);
    /*
    dim3 THREADS_SIZE_L(SIZE_OF_BLOCK, 1);
    dim3 BLOCK_SIZE_L(NX  / THREADS_SIZE_L.x, 1);
    */
    int THREADS_SIZE_L = SIZE_OF_BLOCK;
    int BLOCK_SIZE_L = NX / SIZE_OF_BLOCK;

    /*
    dim3 THREADS_SIZE(SIZE_OF_BLOCK, SIZE_OF_BLOCK);
    dim3 BLOCK_SIZE(NX  / THREADS_SIZE.x, NY / THREADS_SIZE.y);
    */
    dim3 THREADS_SIZE(SIZE_OF_BLOCK, SIZE_OF_BLOCK);
    dim3 BLOCK_SIZE(NX / THREADS_SIZE.x, NY / THREADS_SIZE.y);

    dim3 THREADS_SIZE_test(SIZE_OF_BLOCK_test, SIZE_OF_BLOCK_test, SIZE_OF_BLOCK_test);
    dim3 BLOCK_SIZE_test(NX / THREADS_SIZE_test.x, NY / THREADS_SIZE_test.y, NZ / THREADS_SIZE_test.z);

#endif


    //начальные обработки параметров и начальные кристаллы
#ifdef PREPROCESS

    //Coef_eq << < BLOCK_SIZE, THREADS_SIZE >> > (g, NX, NY, 0);


    //----------------------------CREATION OF CRYSTALS-------------------------//
    printf("%f \n", (int)(0.6 * (NX - 1)) / (double)(NX - 1));
    //create_crystal <<< 1, 1 >>> ((int)(0.6 * (NX - 1)) / (double)(NX - 1), (int)(0.4 * (NY - 1)) / (double)(NY - 1), S_cent_x, S_cent_y, S_left, S_right, S_top, S_bot, S_0, N_cryst, S_cent_x_num, S_cent_y_num, Cryst_age, time);
    create_crystal_3D << < 1, 1 >> > ((int)(0.6 * (NX - 1)) / (double)(NX - 1), (int)(0.3 * (NY - 1)) / (double)(NY - 1), (int)(0.5 * (NX - 1)) / (double)(NX - 1), ALL_PARAMS);
    printf("New_cryst \n");
    //n_img = n_img + 1;
    //N_cryst = N_cryst + 1;

    //create_crystal    <<< 1, 1 >>> ((int)(0.6 * (NX - 1)) / (double)(NX - 1), (int)(0.4 * (NY - 1)) / (double)(NY - 1), S_cent_x, S_cent_y, S_left, S_right, S_top, S_bot, S_0, N_cryst, S_cent_x_num, S_cent_y_num, Cryst_age, time);
    //create_crystal_3D << < 1, 1 >> > ((int)(0.6 * (NX - 1)) / (double)(NX - 1), (int)(0.4 * (NY - 1)) / (double)(NY - 1), 0.0, ALL_PARAMS);

    //k_refresh <<<1, N_cryst + 1 >>> (k_left_arr, k_right_arr, k_top_arr, k_bot_arr, S_left, S_right, S_top, S_bot, NX, NY, S_cent_x_num, S_cent_y_num, dx);
    k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

    Coef_eq_new_cryst_3D << < 1, 1 >> > (d_C, NX, NY, init_value - C_depth_new, N_cryst, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_back_arr, k_forth_arr);


    //----------------------------ITERATIONS ON DEVICE-------------------------//
    printf("start \n");
    printf("%f \n", (double)NX);
    //-Init dCpodX-//
    T = T_start - time_coef * time;
    //M = 4.8 * pow(10, -6) * pow(T, 2) - 8.4 * pow(10, -3) * T + 4.84;
    //C_sat = C_cryst / (exp(10108 / T + 1.16 * (M - 1) - 1.48));
    C_sat = C_cryst / (exp(12900 / T - 0.85 * (M - 1) - 3.80));
    //C_sat = 200;
    //a = fzero(@(x_nd)pi ^ (1 / 2) * x_nd * exp(x_nd ^ 2) * erfc(x_nd) - (C_bound - C_sat) / (C_cryst - C_sat), 0);% вычисление промежуточной величины для аналитического решения;
    D = (exp(-(11.4 * X_H20 + 3.13) / (0.84 * X_H20 + 1) - ((21.4 * X_H20 + 47) / (1.06 * X_H20 + 1)) * (1000) / T));
    D_nd = D / pow(Lx, 2.0) * t_end;


    //k_refresh <<<1, N_cryst + 1 >>> (k_left_arr, k_right_arr, k_top_arr, k_bot_arr, S_left, S_right, S_top, S_bot, NX, NY, S_cent_x_num, S_cent_y_num, dx);
    k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

    //S_bot_refresh << <1, N_cryst + 1 >> > (d_C, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, NX, NY, D_nd, dt, dx, V_bot, dCpoDx_bot, summ, C_sat, C_cryst, a, FLAG_V_limit, FLAGGG_counter);
    //S_top_refresh << <1, N_cryst + 1 >> > (d_C, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, NX, NY, D_nd, dt, dx, V_top, dCpoDx_top, summ, C_sat, C_cryst, a, FLAG_V_limit, FLAGGG_counter);
    //S_left_refresh << <1, N_cryst + 1 >> > (d_C, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, NX, NY, D_nd, dt, dx, V_left, dCpoDx_left, summ, C_sat, C_cryst, a, FLAG_V_limit, FLAGGG_counter);
    //S_right_refresh << <1, N_cryst + 1 >> > (d_C, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, NX, NY, D_nd, dt, dx, V_right, dCpoDx_right, summ, C_sat, C_cryst, a, FLAG_V_limit, FLAGGG_counter);

    S_left_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
    S_right_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

    S_bot_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
    S_top_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

    S_forth_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
    S_back_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

    cudaMemcpy(&p, &summ[0], 1 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("summ[0] = %f \n", p);
    cudaMemcpy(&p, &summ[1], 1 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("summ[1] = %f \n", p);


    //вывод в терминал промежуточной информации
#ifdef PRINT_IN_TERMINAL

    cudaMemcpy(&p, &V_right[0], 1 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("V_right = %f \n", p);

    cudaMemcpy(&p, &dCpoDx_right[0], 1 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("dCpoDx_right[0] = %f \n", p);
    cudaMemcpy(&p, &dCpoDx_left[0], 1 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("dCpoDx_left[0] = %f \n", p);

    cudaMemcpy(&p, &dCpoDx_bot[0], 1 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("dCpoDx_bot[0] = %f \n", p);
    cudaMemcpy(&p, &dCpoDx_top[0], 1 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("dCpoDx_top[0] = %f \n", p);

    cudaMemcpy(&p, &dCpoDx_back[0], 1 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("dCpoDx_back[0] = %f \n", p);
    cudaMemcpy(&p, &dCpoDx_forth[0], 1 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("dCpoDx_forth[0] = %f \n", p);

    for (int k = 0; k < N_cryst; k++) {
        cudaMemcpy(&p, &S_back[k], 1 * sizeof(double), cudaMemcpyDeviceToHost);
        printf("S_back[0] = %f \n", p);
        cudaMemcpy(&p, &S_forth[k], 1 * sizeof(double), cudaMemcpyDeviceToHost);
        printf("S_forth[0] = %f \n", p);
    }
    printf(" \n");



    cudaMemcpy(&p_int, &k_left_arr[0], 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("k_left_arr[0] = %d \n", p_int);

    cudaMemcpy(&p_int, &k_right_arr[0], 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("k_right_arr[0] = %d \n", p_int);


    cudaMemcpy(&p_int, &k_bot_arr[0], 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("k_bot_arr[0] = %d \n", p_int);

    cudaMemcpy(&p_int, &k_top_arr[0], 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("k_top_arr[0] = %d \n", p_int);


    cudaMemcpy(&p_int, &k_forth_arr[0], 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("k_forth_arr[0] = %d \n", p_int);

    cudaMemcpy(&p_int, &k_back_arr[0], 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("k_back_arr[0] = %d \n", p_int);


    cudaMemcpy(&p_int, &S_cent_x_num[0], 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("S_cent_x_num[0] = %d \n", p_int);

    cudaMemcpy(&p_int, &S_cent_y_num[0], 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("S_cent_y_num[0] = %d \n", p_int);

    cudaMemcpy(&p_int, &S_cent_z_num[0], 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("S_cent_z_num[0] = %d \n", p_int);


    cudaMemcpy(&p, &summ[0], 1 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("summ[0] = %f \n", p);
    cudaMemcpy(&p, &summ[1], 1 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("summ[1] = %f \n", p);



    //(int)(0.3 * (NX - 1)) / (double)(NX - 1), (int)(0.6 * (NY - 1)) / (double)(NY - 1), (int)(0.5 * (NX - 1)) / (double)(NX - 1)


    printf("\n N_cryst_dt_razb = %d \n\n", N_cryst_dt_razb);
    //printf("\n FLAGGG_counter  = %d \n\n", FLAGGG_counter[0]);
    //printf("\n C[not real] = %f \n\n", p);
    printf("time               = %f \n", time);
    printf("\n\n\n\n");
#endif



    //Source_eq_y <<<1, N_cryst + 1 >>> (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_bot, dCpoDx_top, S_top, S_bot, S_left, S_right, dCpoDx_right, dCpoDx_left, k_left_arr, k_right_arr, k_top_arr, k_bot_arr);
    //Source_eq_y_3D << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_bot, dCpoDx_top, S_top, S_bot, S_left, S_right, dCpoDx_right, dCpoDx_left, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);


    /*
    cudaMemcpy(h_C_GPU_result, d_C, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(h_C_GPU_result, NX, NY, 777);
    gpuErrchk(cudaPeekAtLastError());
    */

    max_val = init_value;
    i = -1;

#endif

    if (FLAG_all_graphs) {
        graph_number = graph_number + 1;
        cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
        WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, graph_number);
    }


    time_progress = (int)(floor(time * 0));
    WriteInFile_progress(time_progress);

    //Основной цикл
    while (time <= 1) {
        i = i + 1;
        Counter = Counter + 1;

        //дробление времени при появлении новых кристаллов
#ifdef DROB_TIME
        if (FLAG_drob_time) {
            if (FLAG_new_cryst == 1) {
                dt_counter++;
                if (dt_counter == N_dt_razb) {
                    if ((dt * N_dt_razb) < dt_init + 0.000001) {
                        dt = dt * N_dt_razb;
                        N_cryst_dt_razb--;
                    }
                    if (N_cryst_dt_razb > 0) {
                        dt_counter = 0;
                    }
                }
            }
        }
#endif

        //обновление параметров на полушаге
#ifdef REFRESH_PARAMS

#ifdef YAVNO_DIFFUSION
        time = time + dt;
#else
        time = time + dt / 3;
#endif
        arr_time_d[i] = time;

        T = T_start - time_coef * time;
        arr_T_d[i] = T;

        C_sat = C_cryst / (exp(12900 / T - 0.85 * (M - 1) - 3.80));
        D = (exp(-(11.4 * X_H20 + 3.13) / (0.84 * X_H20 + 1) - ((21.4 * X_H20 + 47) / (1.06 * X_H20 + 1)) * (1000) / T));
        D_nd = D / pow(Lx, 2.0) * t_end;

#endif

        //вычисление скорости нуклеации
#ifdef V_NUCLEATION_CALCULATION

        //max_val = 0;
        max_val_x = 0;
        max_val_y = 0;




        //reduceMaxIdx_3D << <1, 1 >> > (d_C, NX * NY * NZ, d_max, d_max_index, gap);

        //cudaMemcpy(&max_val, d_max, sizeof(double), cudaMemcpyDeviceToHost);
#ifdef FIND_ERRORS
        gpuErrchk(cudaPeekAtLastError());
#endif
        //cudaMemcpy(&max_val, d_max, sizeof(double), cudaMemcpyDeviceToHost);

        //max_val = max_val_arr[0];

        // 
        // 
        //reduceMaxIdx << <1, 1 >> > (d_C, NX * NY , d_max, d_max_index, gap);
        //VecMax<< <test_1, test_2 >>> (d_C, d_C_test, NX * NY * NZ);


        //printf(" Rhs[0] = %f \n", max_val);

        /*
        if (FLAG_all_graphs) {
            graph_number = graph_number + 1;
            cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, 100);
        }

        if (FLAG_all_graphs) {
            graph_number = graph_number + 1;
            cudaMemcpy(h_C_GPU_result, d_C_test, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, 101);
        }

        cudaMemcpy(&p_int, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);
        printf("d_max_index  = %d \n", p_int);

        */
        /**/
        
        Max_Sequential_Addressing_Shared_x_optimized << < size / 1024, 1024 >> > (d_C, data_out, size);
        for (int i = 0; i < 2 ; i++) {
            Max_Sequential_Addressing_Shared_x << < size / 1024, 1024 >> > (data_out, size / 1024);
        }
        
        cudaMemcpy(&max_val, &data_out[0], sizeof(double), cudaMemcpyDeviceToHost);
        



#ifdef FIND_ERRORS
        gpuErrchk(cudaPeekAtLastError());
#endif
        max_delta = max_val - C_sat;

        V_nucl = Coef_Nucl_mult * exp(pow((max_delta / Coef_delta_C), Coef_N) * Coef_nucl_c) * exp(-1 / T * (Coef_nucl_T));
        arr_V_nucl_d[i] = V_nucl;

        Numb_cryst_begin = Numb_cryst_begin + V_nucl * dt;


#ifdef PRINT_IN_TERMINAL
        printf("V_nucl = %f \n", V_nucl);
        printf("Numb_cryst_begin = %f \n", Numb_cryst_begin);
        printf("max_delta = %f \n", max_delta);
        printf("d_max = %f \n", max_val);
#endif
#endif


        //создание нового кристалла
#ifdef CREATION_OF_NEW_CRYSTALLS
#ifdef CREATION_OF_NEW_CRYSTALLS_3D
        if (Numb_cryst_begin > n_img) {


            if (FLAG_drob_time) {
                dt_counter = 0;
                FLAG_new_cryst = 1;
                dt = dt / N_dt_razb;
                //printf("\n N_cryst_dt_razb = %d \n\n", N_cryst_dt_razb);
                N_cryst_dt_razb = N_cryst_dt_razb + 1;
                //printf("\n N_cryst_dt_razb = %d \n\n", N_cryst_dt_razb);
            }



            //distance_summer << < BLOCK_SIZE, THREADS_SIZE >> > (distance, NX, NY, N_cryst, S_cent_x, S_cent_y, gap);
            /*
            for (int k = 0; k < NZ ; k++) {
                distance_summer_3D << < BLOCK_SIZE, THREADS_SIZE >> > (&distance[NX * NY * k], NX, NY, NZ, N_cryst, S_cent_x, S_cent_y, S_cent_z, gap, k);
            }
            */

            distance_summer_3D_optimized << < size / 1024, 1024 >> > (distance, NX, NY, NZ, N_cryst, S_cent_x, S_cent_y, S_cent_z, gap);

            //cudaMemcpy(h_C_GPU_result, d_C, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);

            reduceMaxIdx_3D << <1, 1 >> > (d_C, NX * NY * NZ, d_max, d_max_index, gap);

            cudaMemcpy(&max_val, d_max, sizeof(double), cudaMemcpyDeviceToHost);
            /*
            Max_Sequential_Addressing_Shared_x_optimized_gap << < size / 1024, 1024 >> > (d_C, data_out, size, gap);
            for (int i = 0; i < 2; i++) {
                Max_Sequential_Addressing_Shared_x_gap << < size / 1024, 1024 >> > (data_out, size / 1024, gap);
            }
            cudaMemcpy(&max_val, &data_out[0], sizeof(double), cudaMemcpyDeviceToHost);
            */
            //printf("max_val = %f \n", max_val);

            //distance_obnulator_3D << < 1, 1 >> > (distance, d_C, NX, NY, NZ, N_cryst, S_cent_x, S_cent_y, S_cent_z, gap, max_val, eps);

            distance_obnulator_3D_optimized << < size / 1024, 1024 >> > (distance, d_C, NX, NY, NZ, N_cryst, S_cent_x, S_cent_y, S_cent_z, gap, max_val, eps);

            max_val = 0;
            max_val_x = 0;
            max_val_y = 0;


            reduceMaxIdx_3D << <1, 1 >> > (distance, NX * NY * NZ, d_max, d_max_index, gap);

            cudaMemcpy(&max_val_x, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);
            //cudaMemcpy(&max_val, d_max, sizeof(double), cudaMemcpyDeviceToHost);

            /*
            for (int i = 0; i < 3; i++) {
                Max_Sequential_Addressing_Shared_x << < size / 1024, 1024 >> > (d_C, (NX) * (NY) * (NZ));
            }
            cudaMemcpy(&max_val, &d_C[0], sizeof(double), cudaMemcpyDeviceToHost);
            */



            max_val_z = max_val_x / (NX * NY);
            max_val_y = (max_val_x - max_val_z * NX * NY) / NY;
            max_val_x = (max_val_x - max_val_z * NX * NY) - max_val_y * NY;

            //cudaMemcpy(&p, max_val_y_s, 1 * sizeof(double), cudaMemcpyDeviceToHost);


            //cudaMemcpy(&p, max_val_y_s, 1 * sizeof(double), cudaMemcpyDeviceToHost);


            n_img = n_img + 1;
            N_cryst = N_cryst + 1;

            //reduceMaxIdx <<< 1, 1 >>> (d_C, NX* NY, d_max, d_max_index);
            //cudaMemcpy(&max_val, d_max, sizeof(double), cudaMemcpyDeviceToHost);
            //cudaMemcpy(&max_val_x, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);
            //max_val_y = (max_val_x-1) / (NX);
            //max_val_x = max_val_x - max_val_y * (NX);



            max_val_x_s = (double)max_val_x;
            max_val_y_s = (double)max_val_y;
            max_val_z_s = (double)max_val_z;
#ifdef PRINT_IN_TERMINAL
            printf("New_cryst \n");
            printf("\n N_cryst_dt_razb = %d \n\n", N_cryst_dt_razb);
            printf("max_val_x[0] = %d \n", max_val_x);
            //cudaMemcpy(&p, max_val_x_s, 1 * sizeof(double), cudaMemcpyDeviceToHost);
            printf("max_val_z_s[0] = %d \n", max_val_z);
            //cudaMemcpy(&p, max_val_y_s, 1 * sizeof(double), cudaMemcpyDeviceToHost);
            printf("max_val_y_s[0] = %d \n", max_val_y);

            //cudaMemcpy(&p, max_val_x_s, 1 * sizeof(double), cudaMemcpyDeviceToHost);
            printf("max_val_x_s[0] = %f \n", max_val_x_s);
            //cudaMemcpy(&p, max_val_y_s, 1 * sizeof(double), cudaMemcpyDeviceToHost);
            printf("max_val_y_s[0] = %f \n", max_val_y_s);
#endif
            //create_crystal << < 1, 1 >> > (max_val_x_s / (double)(NX - 1), max_val_y_s / (double)(NY - 1), S_cent_x, S_cent_y, S_left, S_right, S_top, S_bot, S_0, N_cryst, S_cent_x_num, S_cent_y_num, Cryst_age, time);

            //create_crystal_3D << < 1, 1 >> > (max_val_x_s / (double)(NX - 1), max_val_y_s / (double)(NY - 1), (double)((int)(0.5 * (NY - 1)))/ (double)(NY - 1), ALL_PARAMS);

            create_crystal_3D << < 1, 1 >> > (max_val_x_s / (double)(NX - 1), max_val_y_s / (double)(NY - 1), max_val_z_s / (double)(NZ - 1), ALL_PARAMS);

            gpuErrchk(cudaPeekAtLastError());
            k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
            Coef_eq_new_cryst_3D << < 1, 1 >> > (d_C, NX, NY, C_sat - C_depth_new, N_cryst, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_back_arr, k_forth_arr);
#ifdef FIND_ERRORS
            gpuErrchk(cudaPeekAtLastError());
#endif
        }
#else
        if (Numb_cryst_begin > n_img) {

            printf("New_cryst \n");
            if (FLAG_drob_time) {
                dt_counter = 0;
                FLAG_new_cryst = 1;
                dt = dt / N_dt_razb;
            }



            //distance_summer << < BLOCK_SIZE, THREADS_SIZE >> > (distance, NX, NY, N_cryst, S_cent_x, S_cent_y, gap);
            distance_summer << < BLOCK_SIZE, THREADS_SIZE >> > (distance, NX, NY, N_cryst, S_cent_x, S_cent_y, gap);


            cudaMemcpy(h_C_GPU_result, d_C, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);

            reduceMaxIdx << <1, 1 >> > (d_C, NX * NY, d_max, d_max_index, gap);
            cudaMemcpy(&max_val, d_max, sizeof(double), cudaMemcpyDeviceToHost);


            printf("New_cryst \n");

            distance_obnulator << < 1, 1 >> > (distance, d_C, NX, NY, N_cryst, S_cent_x, S_cent_y, gap, max_val, eps);



            printf("New_cryst \n");
            max_val = 0;
            max_val_x = 0;
            max_val_y = 0;

            reduceMaxIdx << <1, 1 >> > (distance, NX * NY, d_max, d_max_index, gap);

            cudaMemcpy(&max_val_x, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&max_val, d_max, sizeof(double), cudaMemcpyDeviceToHost);

            max_val_y = max_val_x / NY;
            max_val_x = max_val_x - max_val_y * NY;




            printf("New_cryst \n");
            n_img = n_img + 1;
            N_cryst = N_cryst + 1;

            //reduceMaxIdx <<< 1, 1 >>> (d_C, NX* NY, d_max, d_max_index);
            //cudaMemcpy(&max_val, d_max, sizeof(double), cudaMemcpyDeviceToHost);
            //cudaMemcpy(&max_val_x, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);
            //max_val_y = (max_val_x-1) / (NX);
            //max_val_x = max_val_x - max_val_y * (NX);



            max_val_x_s = (double)max_val_x;
            max_val_y_s = (double)max_val_y;

            //create_crystal << < 1, 1 >> > (max_val_x_s / (double)(NX - 1), max_val_y_s / (double)(NY - 1), S_cent_x, S_cent_y, S_left, S_right, S_top, S_bot, S_0, N_cryst, S_cent_x_num, S_cent_y_num, Cryst_age, time);

            //create_crystal_3D << < 1, 1 >> > (max_val_x_s / (double)(NX - 1), max_val_y_s / (double)(NY - 1), (double)((int)(0.5 * (NY - 1))) / (double)(NY - 1), ALL_PARAMS);

            create_crystal_3D << < 1, 1 >> > (max_val_x_s / (double)(NX - 1), max_val_y_s / (double)(NY - 1), 0.0, ALL_PARAMS);

            gpuErrchk(cudaPeekAtLastError());
            k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
            Coef_eq_new_cryst_3D << < 1, 1 >> > (d_C, NX, NY, C_sat - C_depth_new, N_cryst, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_back_arr, k_forth_arr);
        }
#endif

#endif

#ifdef FIND_ERRORS
        gpuErrchk(cudaPeekAtLastError());
#endif

#ifdef PRINT_IN_TERMINAL
        printf("\n\n \n");
        printf("time = %f \n", time);
        printf("\n\n \n");
        printf("N_cryst = %d \n", N_cryst + 1);
        printf("C_sat = %f \n", C_sat);
#endif


#ifdef YAVNO_DIFFUSION


        Crystals_growth_3D << < 1, N_cryst + 1 >> > (S_top, S_bot, S_left, S_right, S_back, S_forth, V_top, V_bot, V_left, V_right, V_back, V_forth, dt);
        k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

        //diff_func << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (qx, qy, qz, d_C);
        //diff_func_2 << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (Qx, Qy, Qz, qx, qy, qz);



        diff_x << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (qx, d_C);
        diff_y << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (qy, d_C);
        diff_z << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (qz, d_C);

        diff_x << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (Qx, qx);
        diff_y << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (Qy, qy);
        diff_z << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (Qz, qz);


        Yavno_diffusion_3D << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (d_C, d_C_old, D_nd, dx, dy, dz, Qx, Qy, Qz, dt);

        //bounday_cond_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);

        bounday_cond_back_y_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);
        bounday_cond_up_z_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);
        bounday_cond_back_z_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);

        bounday_cond_up_x_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);
        bounday_cond_back_x_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);
        bounday_cond_up_y_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);





        S_left_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
        S_right_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

        S_bot_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
        S_top_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

        S_forth_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
        S_back_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

        Source_eq_x_3D << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_left, dCpoDx_right, S_right, S_left, S_bot, S_top, dCpoDx_top, dCpoDx_bot, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);

#else




        //<<<x>>>
#ifdef X_DIFFUSION

//int k = 0;


        Source_eq_x_3D << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_left, dCpoDx_right, S_right, S_left, S_bot, S_top, dCpoDx_top, dCpoDx_bot, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);
        //set to zero coefficients, for peeodical progonka

        if (FLAG_all_graphs) {
            graph_number = graph_number + 1;
            cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, graph_number);
        }

#ifdef FIND_ERRORS
        gpuErrchk(cudaPeekAtLastError());
#endif
        for (int k = 0; k < limit_3D; k++) {
            //refresh Rhs
            Rhs_GPU_diff_x_3D << < BLOCK_SIZE, THREADS_SIZE >> > (&Rhs[(NX + 2) * (NY + 2) * k], &d_C[(NX) * (NY)*k], NX + 2, NY + 2, dt);

            //refresh Ld Dd Ud
            Renewer_Ld_Dd_Ud << < BLOCK_SIZE, THREADS_SIZE >> > (&Ld[(NX + 2) * (NY + 2) * k], &Dd[(NX + 2) * (NY + 2) * k], &Ud[(NX + 2) * (NY + 2) * k], NX + 2, NY + 2, -dt / 3 * D_nd / dy / dy, 2 * dt / 3 * D_nd / dy / dy + 1);

            //Coef_a_zeros_y_3D << < 1, N_cryst + 1 >> > (Ud, NX, NY, 0, N_cryst, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_back_arr, k_forth_arr, S_cent_x_num);

        }
        /*
        for (int i = 0; i < 3; i++) {
            Max_Sequential_Addressing_Shared_x << < size / 1024, 1024 >> > (d_C, (NX) * (NY) * (NZ));
        }
        cudaMemcpy(&max_val, &d_C[0], sizeof(double), cudaMemcpyDeviceToHost);
        */
        //printf("d_max = %f \n", max_val);


        Coef_a_zeros_x_3D << < 1, N_cryst + 1 >> > (Ud, NX, NY, 0, N_cryst, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_back_arr, k_forth_arr, S_cent_x_num);
        for (int k = 0; k < limit_3D; k++) {

            Coef_eq_i_1_j_0 << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Dd[(NX + 2) * (NY + 2) * k], NX + 2, NY + 2, 1);
            Coef_eq_i_1_j_0 << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Ud[(NX + 2) * (NY + 2) * k], NX + 2, NY + 2, 0);

            Coef_eq_i_1_j_1 << <  BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Ld[(NX + 2) * (NY + 2) * k + (NX + 2) * (NY)], NX + 2, NY + 2, -D_nd / dx);
            Coef_eq_i_1_j_1 << <  BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Dd[(NX + 2) * (NY + 2) * k + (NX + 2) * (NY)], NX + 2, NY + 2, D_nd / dx);

            Coef_eq << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Rhs[(NX + 2) * (NY + 2) * k + (NX + 2) * (NY + 2 - 1) + 1], NX + 2, NY + 2, 0);
            Coef_eq << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Rhs[(NX + 2) * (NY + 2) * k + 1], NX + 2, NY + 2, 0);

            Thomas(&d_C[(NX) * (NY)*k], NX, NY, &Ld[(NX + 2) * (NY + 2) * k], &Dd[(NX + 2) * (NY + 2) * k], &Ud[(NX + 2) * (NY + 2) * k], &Rhs[(NX + 2) * (NY + 2) * k], &x[NX * (NY + 2) * k], 0, SIZE_OF_BLOCK);
            Copy_rev << < BLOCK_SIZE, THREADS_SIZE >> > (&d_C[(NX) * (NY)*k], &x[NX * (NY + 2) * k], NX, NY);
        }


#ifdef FIND_ERRORS
        gpuErrchk(cudaPeekAtLastError());
#endif
        //обновление границ
        //k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
        //k_refresh << <1, N_cryst + 1 >> > (k_left_arr, k_right_arr, k_top_arr, k_bot_arr, S_left, S_right, S_top, S_bot, NX, NY, S_cent_x_num, S_cent_y_num, dx);
        //S_left_refresh << <1, N_cryst + 1 >> > (d_C, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, NX, NY, D_nd, dt, dx, V_left, dCpoDx_left, summ, C_sat, C_cryst, a, FLAG_V_limit, FLAGGG_counter);
        //S_right_refresh << <1, N_cryst + 1 >> > (d_C, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, NX, NY, D_nd, dt, dx, V_right, dCpoDx_right, summ, C_sat, C_cryst, a, FLAG_V_limit, FLAGGG_counter);


        S_left_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
        S_right_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);


        //обновление чего-то

        //Source_eq_y << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_bot, dCpoDx_top, S_top, S_bot, S_left, S_right, dCpoDx_right, dCpoDx_left, k_left_arr, k_right_arr, k_top_arr, k_bot_arr);
        //Source_eq_x << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_left, dCpoDx_right, S_right, S_left, S_bot, S_top, dCpoDx_top, dCpoDx_bot, k_left_arr, k_right_arr, k_top_arr, k_bot_arr);
        //Source_eq_x_3D << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_left, dCpoDx_right, S_right, S_left, S_bot, S_top, dCpoDx_top, dCpoDx_bot, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);


#endif

        if (FLAG_all_graphs) {
            graph_number = graph_number + 1;
            cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, graph_number);
        }

        //рост кристаллов по обновлённым данным
#ifdef CRYSTAL_GROWTH
        Crystals_growth_3D << < 1, N_cryst + 1 >> > (S_top, S_bot, S_left, S_right, S_back, S_forth, V_top, V_bot, V_left, V_right, V_back, V_forth, dt / 3);
        k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
#endif

#ifdef FIND_ERRORS
        gpuErrchk(cudaPeekAtLastError());
#endif

        //обновление параметров на полушаге
#ifdef REFRESH_PARAMS

        time = time + dt / 3;
        arr_time_d[i] = time;

        T = T_start - time_coef * time;
        arr_T_d[i] = T;

        C_sat = C_cryst / (exp(12900 / T - 0.85 * (M - 1) - 3.80));
        D = (exp(-(11.4 * X_H20 + 3.13) / (0.84 * X_H20 + 1) - ((21.4 * X_H20 + 47) / (1.06 * X_H20 + 1)) * (1000) / T));
        D_nd = D / pow(Lx, 2.0) * t_end;

#endif

#ifdef FIND_ERRORS
        gpuErrchk(cudaPeekAtLastError());
#endif

        //<<<y>>>
#ifdef Y_DIFFUSION

#ifdef FIND_ERRORS
        gpuErrchk(cudaPeekAtLastError());
#endif

        Source_eq_y_3D << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_bot, dCpoDx_top, S_top, S_bot, S_left, S_right, dCpoDx_right, dCpoDx_left, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);

        if (FLAG_all_graphs) {
            graph_number = graph_number + 1;
            cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, graph_number);
        }

        for (int k = 0; k < limit_3D; k++) {
            //printf("k = %d \n", k);
            Rhs_GPU_diff_y << < BLOCK_SIZE, THREADS_SIZE >> > (&Rhs[(NX + 2) * (NY + 2) * k], &d_C[(NX) * (NY)*k], divQ, g, NX + 2, NY + 2, dt);

            Renewer_Ld_Dd_Ud << < BLOCK_SIZE, THREADS_SIZE >> > (&Ld[(NX + 2) * (NY + 2) * k], &Dd[(NX + 2) * (NY + 2) * k], &Ud[(NX + 2) * (NY + 2) * k], NX + 2, NY + 2, -dt / 3 * D_nd / dy / dy, 2 * dt / 3 * D_nd / dy / dy + 1);

            //Coef_a_zeros_y << < 1, N_cryst + 1 >> > (&Ud[(NX + 2) * (NY + 2) * 0], NX, NY, 0, N_cryst, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, S_cent_x_num);


        }

        Coef_a_zeros_y_3D << < 1, N_cryst + 1 >> > (Ud, NX, NY, 0, N_cryst, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_back_arr, k_forth_arr, S_cent_y_num);

        for (int k = 0; k < limit_3D; k++) {

            Coef_eq_i_1_j_0 << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Dd[(NX + 2) * (NY + 2) * k], NX + 2, NY + 2, 1);
            Coef_eq_i_1_j_0 << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Ud[(NX + 2) * (NY + 2) * k], NX + 2, NY + 2, 0);

            Coef_eq_i_1_j_1 << <  BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Ld[(NX + 2) * (NY + 2) * k + (NX + 2) * (NY)], NX + 2, NY + 2, -D_nd / dx);
            Coef_eq_i_1_j_1 << <  BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Dd[(NX + 2) * (NY + 2) * k + (NX + 2) * (NY)], NX + 2, NY + 2, D_nd / dx);

            Coef_eq << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Rhs[(NX + 2) * (NY + 2) * k + (NX + 2) * (NY + 2 - 1) + 1], NX + 2, NY + 2, 0);
            Coef_eq << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Rhs[(NX + 2) * (NY + 2) * k + 1], NX + 2, NY + 2, 0);

            Thomas(&d_C[(NX) * (NY)*k], NX, NY, &Ld[(NX + 2) * (NY + 2) * k], &Dd[(NX + 2) * (NY + 2) * k], &Ud[(NX + 2) * (NY + 2) * k], &Rhs[(NX + 2) * (NY + 2) * k], &x[NX * (NY + 2) * k], 0, SIZE_OF_BLOCK);
            Copy << < BLOCK_SIZE, THREADS_SIZE >> > (&d_C[(NX) * (NY)*k], &x[NX * (NY + 2) * k], NX, NY);

        }


        //обновление границ  
        //k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
        //k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
        //S_bot_refresh << <1, N_cryst + 1 >> > (d_C, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, NX, NY, D_nd, dt/3, dx, V_bot, dCpoDx_bot, summ, C_sat, C_cryst, a, FLAG_V_limit, FLAGGG_counter);
        //S_top_refresh << <1, N_cryst + 1 >> > (d_C, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, NX, NY, D_nd, dt / 3, dx, V_top, dCpoDx_top, summ, C_sat, C_cryst, a, FLAG_V_limit, FLAGGG_counter);

        S_bot_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
        S_top_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

        //обновление чего-то
        //Source_eq_y_3D << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_bot, dCpoDx_top, S_top, S_bot, S_left, S_right, dCpoDx_right, dCpoDx_left, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);

        //Source_eq_x << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_left, dCpoDx_right, S_right, S_left, S_bot, S_top, dCpoDx_top, dCpoDx_bot, k_left_arr, k_right_arr, k_top_arr, k_bot_arr);
        //Source_eq_y <<<1, N_cryst + 1 >>> (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_bot, dCpoDx_top, S_top, S_bot, S_left, S_right, dCpoDx_right, dCpoDx_left, k_left_arr, k_right_arr, k_top_arr, k_bot_arr);
        if (FLAG_all_graphs) {
            graph_number = graph_number + 1;
            cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, graph_number);
        }

        //отлов ошибок
#ifdef FIND_ERRORS
        gpuErrchk(cudaPeekAtLastError());
#endif

#endif

        //рост кристаллов по обновлённым данным
#ifdef CRYSTAL_GROWTH
        Crystals_growth_3D << < 1, N_cryst + 1 >> > (S_top, S_bot, S_left, S_right, S_back, S_forth, V_top, V_bot, V_left, V_right, V_back, V_forth, dt / 3);
        k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
#endif

        //обновление параметров на полушаге
#ifdef REFRESH_PARAMS

        time = time + dt / 3;
        arr_time_d[i] = time;

        T = T_start - time_coef * time;
        arr_T_d[i] = T;

        C_sat = C_cryst / (exp(12900 / T - 0.85 * (M - 1) - 3.80));
        D = (exp(-(11.4 * X_H20 + 3.13) / (0.84 * X_H20 + 1) - ((21.4 * X_H20 + 47) / (1.06 * X_H20 + 1)) * (1000) / T));
        D_nd = D / pow(Lx, 2.0) * t_end;

#endif

        //<<<z>>>
#ifdef Z_DIFFUSION
                                    // (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_left, dCpoDx_right, S_right, S_left, S_bot, S_top, dCpoDx_top, dCpoDx_bot, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);
                                     //(d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_bot , dCpoDx_top  , S_top  , S_bot , S_left, S_right, dCpoDx_right, dCpoDx_left, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);
        Source_eq_z_3D << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_back, dCpoDx_forth, S_forth, S_back, S_bot, S_top, dCpoDx_top, dCpoDx_bot, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);

        if (FLAG_all_graphs) {
            graph_number = graph_number + 1;
            cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, graph_number);
        }

        for (int k = 0; k < limit_3D; k++) {

            Rhs_GPU_diff_z_3D << < BLOCK_SIZE, THREADS_SIZE >> > (&Rhs[(NX + 2) * (NY + 2) * k], &d_C[k], divQ, g, NX + 2, NY + 2, dt);

            Renewer_Ld_Dd_Ud << < BLOCK_SIZE, THREADS_SIZE >> > (&Ld[(NX + 2) * (NY + 2) * k], &Dd[(NX + 2) * (NY + 2) * k], &Ud[(NX + 2) * (NY + 2) * k], NX + 2, NY + 2, -dt / 3 * D_nd / dy / dy, 2 * dt / 3 * D_nd / dy / dy + 1);
        }

        Coef_a_zeros_z_3D << < 1, N_cryst + 1 >> > (Ud, NX, NY, 0, N_cryst, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_back_arr, k_forth_arr, S_cent_z_num);

        for (int k = 0; k < limit_3D; k++) {
            Coef_eq_i_1_j_0 << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Dd[(NX + 2) * (NY + 2) * k], NX + 2, NY + 2, 1);
            Coef_eq_i_1_j_0 << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Ud[(NX + 2) * (NY + 2) * k], NX + 2, NY + 2, 0);

            Coef_eq_i_1_j_1 << <  BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Ld[(NX + 2) * (NY + 2) * k + (NX + 2) * (NY)], NX + 2, NY + 2, -D_nd / dx);
            Coef_eq_i_1_j_1 << <  BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Dd[(NX + 2) * (NY + 2) * k + (NX + 2) * (NY)], NX + 2, NY + 2, D_nd / dx);

            Coef_eq << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Rhs[(NX + 2) * (NY + 2) * k + (NX + 2) * (NY + 2 - 1) + 1], NX + 2, NY + 2, 0);
            Coef_eq << < BLOCK_SIZE_L, THREADS_SIZE_L >> > (&Rhs[(NX + 2) * (NY + 2) * k + 1], NX + 2, NY + 2, 0);

            Thomas(&d_C[k], NX, NY, &Ld[(NX + 2) * (NY + 2) * k], &Dd[(NX + 2) * (NY + 2) * k], &Ud[(NX + 2) * (NY + 2) * k], &Rhs[(NX + 2) * (NY + 2) * k], x, 0, SIZE_OF_BLOCK);
            //--Copy_rev_3D << < BLOCK_SIZE, THREADS_SIZE >> > (&d_C[(NX) * (NY)*k], x, NX, NY);
            Copy_z_3D << < BLOCK_SIZE, THREADS_SIZE >> > (&d_C[k], x, NX, NY);
        }

        //обновление границ  
        //k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
        //S_left_refresh <<<1, N_cryst + 1 >>> (d_C, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, NX, NY, D_nd, dt / 3, dx, V_left, dCpoDx_left, summ, C_sat, C_cryst, a, FLAG_V_limit, FLAGGG_counter);
        //S_right_refresh <<<1, N_cryst + 1 >>> (d_C, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, NX, NY, D_nd, dt / 3, dx, V_right, dCpoDx_right, summ, C_sat, C_cryst, a, FLAG_V_limit, FLAGGG_counter);

        S_back_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
        S_forth_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

        //обновление чего-то
        //Source_eq_y <<<1, N_cryst + 1 >>> (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_bot, dCpoDx_top, S_top, S_bot, S_left, S_right, dCpoDx_right, dCpoDx_left, k_left_arr, k_right_arr, k_top_arr, k_bot_arr);
        //Source_eq_x_3D << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_left, dCpoDx_right, S_right, S_left, S_bot, S_top, dCpoDx_top, dCpoDx_bot, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);

        if (FLAG_all_graphs) {
            graph_number = graph_number + 1;
            cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, graph_number);
        }

        //отлов ошибок
#ifdef FIND_ERRORS
        gpuErrchk(cudaPeekAtLastError());
#endif

#endif


        //рост кристаллов по обновлённым данным
#ifdef CRYSTAL_GROWTH
        Crystals_growth_3D << < 1, N_cryst + 1 >> > (S_top, S_bot, S_left, S_right, S_back, S_forth, V_top, V_bot, V_left, V_right, V_back, V_forth, dt / 3);
        k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
#endif

#endif

#ifdef WRITE_PROGRESS
        if (fmod(Counter, Nt / Graph_num) == 0) {
            time_progress = (int)(floor(time * 100));
            WriteInFile_progress(time_progress);
        }
#endif

        //запись промежуточной информации в файлы
#ifdef PRINT_IN_FILES
        if (fmod(Counter, Nt / Graph_num) == 0) {

            printf("time = %f \n", time);
            printf("D_nd = %f \n", D_nd);
            printf("N_cryst = %d \n", N_cryst + 1);
            printf("max_val = %f \n\n", max_val);
            printf("C_sat = %f \n", C_sat);
            printf("dt = %f \n", dt);

            for (int N_count = 0; N_count <= N_cryst; N_count++) {
                //printf("        Crystal# %d \n", N_count + 1);
                //printf("S_left = %f \n", S_left[N_count]);
                //printf("S_bot = %.10f \n", S_bot[N_count]);
                //printf("S_right = %.10f \n", S_right[N_count]);
                //printf("S_top = %.10f \n\n", S_top[N_count]);

            }
            graph_number = graph_number + 1;
            cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
            //WriteInFile(h_C_GPU_result, NX, NY, graph_number);
            WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, graph_number);
            //WriteInFile(distance, NX, NY, 0);
            //cudaMemcpy(h_C_GPU_result, arr_time_d, Nt * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile(arr_time_d, Nt, 1, 1);

            // array V_nucl
            //cudaMemcpy(h_C_GPU_result, arr_V_nucl_d, Nt * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile(arr_V_nucl_d, Nt, 1, 2);

            // array T_nucl
            //cudaMemcpy(h_C_GPU_result, arr_T_d, Nt * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile(arr_T_d, Nt, 1, 3);


            cudaMemcpy(h_C_GPU_result, S_left, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10004);

            cudaMemcpy(h_C_GPU_result, S_right, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10005);

            cudaMemcpy(h_C_GPU_result, S_top, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10006);

            cudaMemcpy(h_C_GPU_result, S_bot, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10007);

            cudaMemcpy(h_C_GPU_result, S_back, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10008);

            cudaMemcpy(h_C_GPU_result, S_forth, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10009);

            cudaMemcpy(h_C_GPU_result, Cryst_age, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
            WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 8);


            //WriteInFile(S_left, N_cryst, 1, 4);
            //WriteInFile(S_right, N_cryst, 1, 5);
            //WriteInFile(S_top, N_cryst, 1, 6);
            //WriteInFile(S_bot, N_cryst, 1, 7);
        }
#endif


        //отлов ошибок
#ifdef FIND_ERRORS
        gpuErrchk(cudaPeekAtLastError());
#endif

        //обновление предыдущего слоя по времени
        //cudaMemcpy(d_C_old, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToDevice);
        d_C_old = d_C;
    }


    //последний вывод на экран и запись в файлы
#ifdef LAST_PRINT
    printf("\n\n");
    printf("time = %f \n", time);
    printf("D_nd = %f \n", D_nd);
    printf("N_cryst = %d \n\n", N_cryst + 1);

    for (int N_count = 0; N_count <= N_cryst; N_count++) {
        //printf("        Crystal# %d \n", N_count + 1);
        //printf("S_left = %f \n", S_left[N_count]);
        //printf("S_bot = %.10f \n", S_bot[N_count]);
        //printf("S_right = %.10f \n", S_right[N_count]);
        //printf("S_top = %.10f \n\n", S_top[N_count]);
    }
    graph_number = graph_number + 1;
    cudaMemcpy(h_C_GPU_result, d_C, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(h_C_GPU_result, NX, NY, graph_number);

    WriteInFile(h_C_GPU_result, NX, NY, 777);

    cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, 999);
    //WriteInFile(distance, NX, NY, 0);
    //cudaMemcpy(h_C_GPU_result, arr_time_d, Nt * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(arr_time_d, Nt, 1, 1);

    // array V_nucl
    //cudaMemcpy(h_C_GPU_result, arr_V_nucl_d, Nt * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(arr_V_nucl_d, Nt, 1, 2);

    // array T_nucl
    //cudaMemcpy(h_C_GPU_result, arr_T_d, Nt * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(arr_T_d, Nt, 1, 3);

    //WriteInFile(S_left, N_cryst, 1, 444);
    //WriteInFile(S_right, N_cryst, 1, 555);
    //WriteInFile(S_top, N_cryst, 1, 666);
    //WriteInFile(S_bot, N_cryst, 1, 888);


    cudaMemcpy(&p_int, FLAGGG_counter, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Super_velocity_counter = %d \n", p_int);

    //cudaMemcpy(h_C_GPU_result, distance, NX * NY  * sizeof(double), cudaMemcpyDeviceToHost);
    //WriteInFile(h_C_GPU_result, NX, NY, 0);

    cudaMemcpy(h_C_GPU_result, distance, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, 0);

    // array time
    //cudaMemcpy(h_C_GPU_result, arr_time_d, Nt * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(arr_time_d, (Nt + N_cryst_max * N_dt_razb), 1, 1);

    // array V_nucl
    //cudaMemcpy(h_C_GPU_result, arr_V_nucl_d, Nt * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(arr_V_nucl_d, (Nt + N_cryst_max * N_dt_razb), 1, 2);

    // array T_nucl
    //cudaMemcpy(h_C_GPU_result, arr_T_d, Nt * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(arr_T_d, (Nt + N_cryst_max * N_dt_razb), 1, 3);


    cudaMemcpy(h_C_GPU_result, S_left, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10004);

    cudaMemcpy(h_C_GPU_result, S_right, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10005);

    cudaMemcpy(h_C_GPU_result, S_top, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10006);

    cudaMemcpy(h_C_GPU_result, S_bot, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10007);

    cudaMemcpy(h_C_GPU_result, S_back, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10008);

    cudaMemcpy(h_C_GPU_result, S_forth, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 10009);

    cudaMemcpy(h_C_GPU_result, Cryst_age, (N_cryst + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    WriteInFile(h_C_GPU_result, (N_cryst + 1), 1, 8);
#endif

#ifdef WRITE_PROGRESS
    if (fmod(Counter, Nt / Graph_num) == 0) {
        time_progress = 100;
        WriteInFile_progress(time_progress);
    }
#endif


    //освобождение памяти
#ifdef FREE_MEMORY
// --- Release device memory
    cudaFree(d_C);
    cudaFree(d_C_old);

    free(h_C);
    free(h_C_GPU_result);
#endif


    //заключительная обработка времени
#ifdef CLOCK_END
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    //вывод времени, которое заняло выполнение рпограммы
    printf("Elapsed time : %f ms\n", elapsedTime);
#endif


    return 0;
}