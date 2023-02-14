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


int main() {

	//инициализация
#ifdef INIT_VALUES
	double dt = 1 / (double)(Nt - 1);

	int i = 0;
	int N_cryst = 0;
	double time = 0;
	int Counter = 0;
	int time_progress;
	double max_val = 0;
	int dt_counter = 0;
	double dt_init = dt;
	cudaEvent_t	stop = 0;
	int graph_number = 10;
	float elapsedTime = 0;
	cudaEvent_t start = 0;
	double T = 0, D_nd = 0;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	int max_val_x = 0, max_val_y = 0, max_val_z = 0;
	double max_val_x_s = 0, max_val_y_s = 0, max_val_z_s = 0;

	double* max_val_arr = (double*)calloc(1, sizeof(double));									assert(max_val_arr != NULL);
	double* h_C = (double*)calloc(NX * NY * NZ, sizeof(double));								assert(h_C != NULL);
	double* h_C_old = (double*)calloc(NX * NY * NZ, sizeof(double));							assert(h_C_old != NULL);
	double* h_C_GPU_result = (double*)calloc((NX + 2) * (NY + 2) * (NZ + 2), sizeof(double));	assert(h_C_GPU_result != NULL);

	Initialize_3D(h_C, NX, NY, NZ, init_value);


	double pp = 0, p = 0;
	int p_int = 0;;

	double* arr_dt = (double*)malloc(N_dt_razb * sizeof(double));       assert(arr_dt != NULL);

	//Обработка режима увеличенных кристаллов
	if (FLAG_S_0_increaser && (S_0 < dx)) {
		S_0 = dx;
	}

	for (int j = 1; j <= N_dt_razb; j++) {
		arr_dt[j - 1] = 1 * (1.0 / (double)N_dt_razb) * dt;
	}

	//Инициализация массивов
	double* d_C;			cudaMalloc((void**)&d_C, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));			assert(d_C != NULL);
	double* d_C_old;		cudaMalloc((void**)&d_C_old, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));		assert(d_C_old != NULL);

	double* d_max;			cudaMalloc((void**)&d_max, 2 * sizeof(double));										assert(d_max != NULL);
	int* d_max_index;		cudaMalloc((void**)&d_max_index, 2 * sizeof(int));									assert(d_max_index != NULL);

	double* g;              cudaMalloc((void**)&g, 1 * sizeof(double));											assert(g != NULL);
	double* divQ;           cudaMalloc((void**)&divQ, 1 * sizeof(double));										assert(divQ != NULL);
	double* qx;             cudaMalloc((void**)&qx, (NX) * (NY) * (NZ) * sizeof(double));						assert(qx != NULL);
	double* qy;             cudaMalloc((void**)&qy, (NX) * (NY) * (NZ) * sizeof(double));						assert(qy != NULL);
	double* qz;             cudaMalloc((void**)&qz, (NX) * (NY) * (NZ) * sizeof(double));						assert(qz != NULL);
	double* Qx;             cudaMalloc((void**)&Qx, (NX) * (NY) * (NZ) * sizeof(double));						assert(Qx != NULL);
	double* Qy;             cudaMalloc((void**)&Qy, (NX) * (NY) * (NZ) * sizeof(double));						assert(Qy != NULL);
	double* Qz;             cudaMalloc((void**)&Qz, (NX) * (NY) * (NZ) * sizeof(double));						assert(Qz != NULL);
	double* Ld;             cudaMalloc((void**)&Ld, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));			assert(Ld != NULL);
	double* Dd;             cudaMalloc((void**)&Dd, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));			assert(Dd != NULL);
	double* Ud;             cudaMalloc((void**)&Ud, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));			assert(Ud != NULL);
	double* Rhs;            cudaMalloc((void**)&Rhs, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));			assert(Rhs != NULL);
	double* x;              cudaMalloc((void**)&x, (NX + 2) * (NY + 2) * (NZ + 2) * sizeof(double));			assert(x != NULL);

	double* data_out;       cudaMalloc((void**)&data_out, (size / 1024) * sizeof(double));                     assert(data_out != NULL);
	int* FLAGGG_counter;    cudaMalloc((void**)&FLAGGG_counter, 1 * sizeof(int));								assert(FLAGGG_counter != NULL);

	cudaMemcpy(d_C, h_C_GPU_result, (NX) * (NY) * (NZ) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_old, h_C_GPU_result, (NX) * (NY) * (NZ) * sizeof(double), cudaMemcpyHostToDevice);
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
	int* k_back_arr;        cudaMalloc((void**)&k_back_arr, N_cryst_max * sizeof(int));			assert(k_back_arr != NULL);
	int* k_forth_arr;       cudaMalloc((void**)&k_forth_arr, N_cryst_max * sizeof(int));		assert(k_forth_arr != NULL);

	double* arr_time_d		= (double*)malloc(5 * (Nt + N_cryst_max * N_dt_razb) * sizeof(double));			assert(arr_time_d != NULL);
	double* arr_V_nucl_d	= (double*)malloc(5 * (Nt + N_cryst_max * N_dt_razb) * sizeof(double));			assert(arr_V_nucl_d != NULL);
	double* arr_T_d			= (double*)malloc(5 * (Nt + N_cryst_max * N_dt_razb) * sizeof(double));         assert(arr_T_d != NULL);


	double* S_cent_x;		cudaMalloc((void**)&S_cent_x, N_cryst_max * sizeof(double));        assert(S_cent_x != NULL);
	double* S_cent_y;		cudaMalloc((void**)&S_cent_y, N_cryst_max * sizeof(double));        assert(S_cent_y != NULL);
	double* S_cent_z;		cudaMalloc((void**)&S_cent_z, N_cryst_max * sizeof(double));        assert(S_cent_z != NULL);


	double* S_left;			cudaMalloc((void**)&S_left, N_cryst_max * sizeof(double));          assert(S_left != NULL);
	double* S_right;		cudaMalloc((void**)&S_right, N_cryst_max * sizeof(double));         assert(S_right != NULL);
	double* S_top;			cudaMalloc((void**)&S_top, N_cryst_max * sizeof(double));           assert(S_top != NULL);
	double* S_bot;			cudaMalloc((void**)&S_bot, N_cryst_max * sizeof(double));           assert(S_bot != NULL);
	double* S_back;			cudaMalloc((void**)&S_back, N_cryst_max * sizeof(double));          assert(S_back != NULL);
	double* S_forth;		cudaMalloc((void**)&S_forth, N_cryst_max * sizeof(double));         assert(S_forth != NULL);

	double* Cryst_age;		cudaMalloc((void**)&Cryst_age, N_cryst_max * sizeof(double));       assert(Cryst_age != NULL);

	int* S_cent_x_num;		cudaMalloc((void**)&S_cent_x_num, N_cryst_max * sizeof(int));       assert(S_cent_x_num != NULL);
	int* S_cent_y_num;		cudaMalloc((void**)&S_cent_y_num, N_cryst_max * sizeof(int));       assert(S_cent_y_num != NULL);
	int* S_cent_z_num;		cudaMalloc((void**)&S_cent_z_num, N_cryst_max * sizeof(int));       assert(S_cent_z_num != NULL);

	double* distance;		cudaMalloc((void**)&distance, NX * NY * NZ * sizeof(double));       assert(distance != NULL);

	double* V_left;			cudaMalloc((void**)&V_left, N_cryst_max * sizeof(double));          assert(V_left != NULL);
	double* V_right;		cudaMalloc((void**)&V_right, N_cryst_max * sizeof(double));         assert(V_right != NULL);
	double* V_top;			cudaMalloc((void**)&V_top, N_cryst_max * sizeof(double));           assert(V_top != NULL);
	double* V_bot;			cudaMalloc((void**)&V_bot, N_cryst_max * sizeof(double));           assert(V_bot != NULL);
	double* V_forth;        cudaMalloc((void**)&V_forth, N_cryst_max * sizeof(double));			assert(V_forth != NULL);
	double* V_back;         cudaMalloc((void**)&V_back, N_cryst_max * sizeof(double));          assert(V_back != NULL);

	double* summ;			cudaMalloc((void**)&summ, 2 * sizeof(double));                      assert(summ != NULL);

	double* buffer = (double*)malloc(NX * NY * NZ * sizeof(double));                            assert(buffer != NULL);

	double* d_summ;          cudaMalloc((void**)&d_summ, 2 * sizeof(double));					assert(d_summ != NULL);


	double* dCpoDx_left;        cudaMalloc((void**)&dCpoDx_left, N_cryst_max * sizeof(double));     assert(dCpoDx_left != NULL);
	double* dCpoDx_right;       cudaMalloc((void**)&dCpoDx_right, N_cryst_max * sizeof(double));    assert(dCpoDx_right != NULL);
	double* dCpoDx_top;         cudaMalloc((void**)&dCpoDx_top, N_cryst_max * sizeof(double));      assert(dCpoDx_top != NULL);
	double* dCpoDx_bot;         cudaMalloc((void**)&dCpoDx_bot, N_cryst_max * sizeof(double));      assert(dCpoDx_bot != NULL);
	double* dCpoDx_forth;       cudaMalloc((void**)&dCpoDx_forth, N_cryst_max * sizeof(double));    assert(dCpoDx_forth != NULL);
	double* dCpoDx_back;        cudaMalloc((void**)&dCpoDx_back, N_cryst_max * sizeof(double));     assert(dCpoDx_back != NULL);



	cudaMemcpy(d_C, h_C, NX * NY * NZ * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_old, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToDevice);

	int THREADS_SIZE_L = SIZE_OF_BLOCK;
	int BLOCK_SIZE_L = NX / SIZE_OF_BLOCK;


	dim3 THREADS_SIZE(SIZE_OF_BLOCK, SIZE_OF_BLOCK);
	dim3 BLOCK_SIZE(NX / THREADS_SIZE.x, NY / THREADS_SIZE.y);

	dim3 THREADS_SIZE_test(SIZE_OF_BLOCK_test, SIZE_OF_BLOCK_test, SIZE_OF_BLOCK_test);
	dim3 BLOCK_SIZE_test(NX / THREADS_SIZE_test.x, NY / THREADS_SIZE_test.y, NZ / THREADS_SIZE_test.z);

#endif

	//Инициализация начальныч параметров и начальные кристаллы
#ifdef PREPROCESS


	//----------------------------CREATION OF CRYSTALS-------------------------//
	printf("%f \n", (int)(0.6 * (NX - 1)) / (double)(NX - 1));
	create_crystal_3D << < 1, 1 >> > ((int)(0.6 * (NX - 1)) / (double)(NX - 1), (int)(0.3 * (NY - 1)) / (double)(NY - 1), (int)(0.5 * (NX - 1)) / (double)(NX - 1), ALL_PARAMS);
	printf("New_cryst \n");
	k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
	Coef_eq_new_cryst_3D << < 1, 1 >> > (d_C, NX, NY, init_value - C_depth_new, N_cryst, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_back_arr, k_forth_arr);


	//----------------------------ITERATIONS ON DEVICE-------------------------//
	printf("start \n");
	printf("%f \n", (double)NX);
	T = T_start - time_coef * time;
	//M = 4.8 * pow(10, -6) * pow(T, 2) - 8.4 * pow(10, -3) * T + 4.84;
	//C_sat = C_cryst / (exp(10108 / T + 1.16 * (M - 1) - 1.48));
	C_sat = C_cryst / (exp(12900 / T - 0.85 * (M - 1) - 3.80));
	D = (exp(-(11.4 * X_H20 + 3.13) / (0.84 * X_H20 + 1) - ((21.4 * X_H20 + 47) / (1.06 * X_H20 + 1)) * (1000) / T));
	D_nd = D / pow(Lx, 2.0) * t_end;

	k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

	S_left_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
	S_right_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

	S_bot_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
	S_top_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

	S_forth_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
	S_back_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);

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

	max_val = init_value;
	i = -1;

	if (FLAG_all_graphs) {
		graph_number = graph_number + 1;
		cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
		WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, graph_number);
	}

	time_progress = (int)(floor(time * 0));
	WriteInFile_progress(time_progress);

#endif


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
		for (int i = 0; i < 2; i++) {
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
#ifdef CREATION_OF_NEW_CRYSTALLS_3D
		if (Numb_cryst_begin > n_img) {

			if (FLAG_drob_time) {
				dt_counter = 0;
				FLAG_new_cryst = 1;
				dt = dt / N_dt_razb;
				N_cryst_dt_razb = N_cryst_dt_razb + 1;
			}

			//вычисление матрицы растояний
			distance_summer_3D_optimized << < size / 1024, 1024 >> > (distance, NX, NY, NZ, N_cryst, S_cent_x, S_cent_y, S_cent_z, gap);
			
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


			max_val_x_s = (double)max_val_x;
			max_val_y_s = (double)max_val_y;
			max_val_z_s = (double)max_val_z;
#ifdef PRINT_IN_TERMINAL
			printf("New_cryst \n");
			printf("\n N_cryst_dt_razb = %d \n\n", N_cryst_dt_razb);
			printf("max_val_x[0] = %d \n", max_val_x);

			printf("max_val_z_s[0] = %d \n", max_val_z);
			printf("max_val_y_s[0] = %d \n", max_val_y);

			printf("max_val_x_s[0] = %f \n", max_val_x_s);
			printf("max_val_y_s[0] = %f \n", max_val_y_s);
#endif

			create_crystal_3D << < 1, 1 >> > (max_val_x_s / (double)(NX - 1), max_val_y_s / (double)(NY - 1), max_val_z_s / (double)(NZ - 1), ALL_PARAMS);

			gpuErrchk(cudaPeekAtLastError());
			k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
			Coef_eq_new_cryst_3D << < 1, 1 >> > (d_C, NX, NY, C_sat - C_depth_new, N_cryst, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_back_arr, k_forth_arr);
#ifdef FIND_ERRORS
			gpuErrchk(cudaPeekAtLastError());
#endif
		}
		
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


		diff_x << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (qx, d_C);
		diff_y << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (qy, d_C);
		diff_z << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (qz, d_C);

		diff_x << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (Qx, qx);
		diff_y << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (Qy, qy);
		diff_z << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (Qz, qz); 

		Yavno_diffusion_3D << < BLOCK_SIZE_test, THREADS_SIZE_test >> > (d_C, d_C_old, D_nd, dx, dy, dz, Qx, Qy, Qz, dt);

		bounday_cond_back_y_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);
		bounday_cond_up_z_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);
		bounday_cond_back_z_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);

		bounday_cond_up_x_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);
		bounday_cond_back_x_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);
		bounday_cond_up_y_3D << < BLOCK_SIZE, THREADS_SIZE >> > (d_C);

		S_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
		Crystals_growth_3D << < 1, N_cryst + 1 >> > (S_top, S_bot, S_left, S_right, S_back, S_forth, V_top, V_bot, V_left, V_right, V_back, V_forth, dt);
		k_refresh_3D << <1, N_cryst + 1 >> > (ALL_PARAMS);
		Source_eq_x_3D << <1, N_cryst + 1 >> > (d_C, NX, NY, C_sat, dx, D_nd, dCpoDx_left, dCpoDx_right, S_right, S_left, S_bot, S_top, dCpoDx_top, dCpoDx_bot, k_left_arr, k_right_arr, k_top_arr, k_bot_arr, k_forth_arr, k_back_arr);

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


		//Обработка ошибок
#ifdef FIND_ERRORS
		gpuErrchk(cudaPeekAtLastError());
#endif

		//обновление предыдущего слоя по времени
		d_C_old = d_C;
	}

	//последний вывод на экран и запись в файлы
#ifdef LAST_PRINT
	printf("\n\n");
	printf("time = %f \n", time);
	printf("D_nd = %f \n", D_nd);
	printf("N_cryst = %d \n\n", N_cryst + 1);

	//Выводинформации о кристалах
	/*
	for (int N_count = 0; N_count <= N_cryst; N_count++) {
		printf("        Crystal# %d \n", N_count + 1);
		printf("S_left = %f \n", S_left[N_count]);
		printf("S_bot = %.10f \n", S_bot[N_count]);
		printf("S_right = %.10f \n", S_right[N_count]);
		printf("S_top = %.10f \n\n", S_top[N_count]);
	}
	*/

	graph_number = graph_number + 1;
	cudaMemcpy(h_C_GPU_result, d_C, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
	WriteInFile(h_C_GPU_result, NX, NY, graph_number);

	WriteInFile(h_C_GPU_result, NX, NY, 777);

	cudaMemcpy(h_C_GPU_result, d_C, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
	WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, 999);

	WriteInFile(arr_time_d, Nt, 1, 1);

	// array V_nucl
	WriteInFile(arr_V_nucl_d, Nt, 1, 2);

	// array T_nucl
	WriteInFile(arr_T_d, Nt, 1, 3);


	cudaMemcpy(&p_int, FLAGGG_counter, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	printf("Super_velocity_counter = %d \n", p_int);

	//distance
	cudaMemcpy(h_C_GPU_result, distance, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost);
	WriteInFile_3D(h_C_GPU_result, NX, NY, NZ, 0);

	// array time
	WriteInFile(arr_time_d, (Nt + N_cryst_max * N_dt_razb), 1, 1);

	// array V_nucl
	WriteInFile(arr_V_nucl_d, (Nt + N_cryst_max * N_dt_razb), 1, 2);

	// array T_nucl
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

	//вывод времени, которое ушло на выполнение рпограммы
	printf("Elapsed time : %f ms\n", elapsedTime);
#endif

	return 0;
}