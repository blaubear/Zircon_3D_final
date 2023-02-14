
const int NX = 64;
const int NY = NX;
const int NZ = NY;

int size = NX * NX * NX;

int N_cryst_max   = 2000;
int Nt = 32 * NX * NX;

double T_start = (double)800 + 273;     // начальная температура в кельвинах;
double T_end = (double)770.0811 + 273;     //конечная температура в кельвинах;
double M = 1.2873;
double X_H20 = 3;
double Lx = 0.005;
int N_cryst_dt_razb = 0;
double dx         = 1.0 / (NX - 1);
double dy         = dx;
double dz		  = dx;
double D = 0.1;
double t_end_yrs  = 10000;
double time_coef  = (T_start - T_end);
double t_end      = t_end_yrs * 365 * 24 * 60 * 60.0; // окончание по времени;


//минимальный радиус кристала (обезразмеренный)
double S_0 =  5.e-6 / Lx;


int n_img				= 1;
double V_nucl           = 0;
double max_delta		= 0;
double Numb_cryst_begin = 0;

double a = 5.e-5; // меньше - больше ограничения, чем меньше, тем медленнее перешагивает

int gap = NX/32 + 2;

double eps = 0.01; //чем больше - тем больше влияние растояния

int N_dt_razb = 4;

int FLAG_new_cryst = 0; //help flag
int FLAG_k_top_k_bot = 0;



int FLAG_g = 0;

int FLAG_V_limit = 1;

int Graph_num = 5;

int SIZE_OF_BLOCK = 32;

int SIZE_OF_BLOCK_test = 8;

double UNO            = 1;

double C_depth_new    = 1;

int FLAG_create_cryst = 1; 

int FLAG_all_graphs = 0;

int FLAG_1d_test = 0;


int FLAG_drob_time = 1;

double C_cryst = 490000;


double C_sat = C_cryst / (exp(12900 / T_start - 0.85 * (M - 1) - 3.80));
double init_value = C_sat;

double Coef_Nucl_mult = Lx * Lx * Lx * t_end_yrs * 5.e+4;
double Coef_delta_C = 4.7;    //    чем больше, тем более плоский график, чем меньше, тем быстрее наступит пик, начало остаётся примерно такое же
double Coef_N = 5.53; //    сдвиг пика влево и вверх при ывеличении, а также уменьшение значений меньше 0
double Coef_nucl_c = 0.06e-2; // уменьшение - снижает значимость разницы концентраций
double Coef_nucl_T = 0.05e+4;  // увеличение - опускает всю кривую, а именно начальную точку, отдаляет пик от начала
//чтобы увеличить пик - нужно увеличить N и delta_C ?




int FLAG_S_0_increaser = 0;

#define YAVNO_DIFFUSION

#define X_DIFFUSION
#define Y_DIFFUSION
#define Z_DIFFUSION

#define PRINT_IN_TERMINAL

#define FIND_ERRORS

#define PRINT_IN_FILES

#define V_NUCLEATION_CALCULATION

#define DROB_TIME

#define CRYSTAL_GROWTH
#define REFRESH_PARAMS
#define CREATION_OF_NEW_CRYSTALLS
#define LAST_PRINT
#define PREPROCESS
#define FREE_MEMORY
#define CLOCK_END
#define INIT_VALUES
#define CREATION_OF_NEW_CRYSTALLS_3D
#define WRITE_PROGRESS


int limit_3D = NX;

#define XXX
#define YYY
#define ZZZ


//int* k_back_arr, int* k_forth_arr 
#define ALL_PARAMS					 d_C,	      S_cent_x,			S_cent_y,			S_cent_z,			S_left,			S_right,			S_top,			S_bot,			S_back,			S_forth,		S_0,	 N_cryst,		S_cent_x_num,		S_cent_y_num,		S_cent_z_num,			Cryst_age,		  time,			k_left_arr,			k_right_arr,		k_top_arr,		k_bot_arr,		k_back_arr,			k_forth_arr,		dx,				dCpoDx_left,			dCpoDx_right,			dCpoDx_top,			dCpoDx_bot,				dCpoDx_forth,			dCpoDx_back,		 V_left,		 V_right,			V_top,			V_bot,			V_forth,			V_back,			D_nd,			dt,			FLAGGG_counter,			summ,			C_sat,			a,			C_cryst,		FLAG_V_limit
#define ALL_PARAMS_TYPES	 double* d_C, double* S_cent_x, double*	S_cent_y,  double*	S_cent_z, double*	S_left, double* S_right, double*	S_top, double*	S_bot, double*	S_back, double*	S_forth, double S_0, int N_cryst, int*	S_cent_x_num, int*	S_cent_y_num, int*	S_cent_z_num,	double* Cryst_age, double time, int*	k_left_arr, int*	k_right_arr, int*	k_top_arr, int* k_bot_arr, int* k_back_arr, int*	k_forth_arr, double dx,  double*	dCpoDx_left, double*	dCpoDx_right,  double*	dCpoDx_top, double* dCpoDx_bot,  double*	dCpoDx_forth, double*	dCpoDx_back, double* V_left, double* V_right, double*	V_top, double*	V_bot, double*	V_forth, double*	V_back, double	D_nd, double	dt, int*	FLAGGG_counter, double* summ, double	C_sat, double	a, double	C_cryst, int	FLAG_V_limit
//LV - 3
/*
double T_start    = (double)767 + 273;     // начальная температура в кельвинах;
double T_end      = (double)749.0599 + 273;     //конечная температура в кельвинах;
double M = 1.5046;
double X_H20 = 6;

double eps = 0.00001;
double Lx         = 0.1;

double Coef_Nucl_mult   = Lx * Lx * t_end_yrs * 1.3e+1;
double Coef_delta_C     = 0.75;    //    чем больше, тем более плоский график, чем меньше, тем быстрее наступит пик, начало остаётся примерно такое же
double Coef_N           = 2.8; //    сдвиг пика влево и вверх при ывеличении, а также уменьшение значений меньше 0
double Coef_nucl_c      = 0.06e-2; // уменьшение - снижает значимость разницы концентраций
double Coef_nucl_T      = 0.4e+4;  // увеличение - опускает всю кривую, а именно начальную точку, отдаляет пик от начала
//чтобы увеличить пик - нужно увеличить N и delta_C ?
*/



//HRT-A
/*
double T_start    = (double)863 + 273;     // начальная температура в кельвинах;
double T_end      = (double)855.1239 + 273;     //конечная температура в кельвинах;
double M = 1.29;
double X_H20 = 3;
double Lx         = 0.2;

double Coef_Nucl_mult   = Lx * Lx * t_end_yrs * 0.09e+1;
double Coef_delta_C     = 0.45;    //    чем больше, тем более плоский график, чем меньше, тем быстрее наступит пик, начало остаётся примерно такое же
double Coef_N           = 2.3; //    сдвиг пика влево и вверх при ывеличении, а также уменьшение значений меньше 0
double Coef_nucl_c      = 0.06e-2; // уменьшение - снижает значимость разницы концентраций
double Coef_nucl_T      = 0.4e+4;  // увеличение - опускает всю кривую, а именно начальную точку, отдаляет пик от начала
//чтобы увеличить пик - нужно увеличить N и delta_C ?
*/



//MFT-1 ?
/* 
double T_start    = (double)800 + 273;     // начальная температура в кельвинах;
double T_end      = (double)770.0811 + 273;     //конечная температура в кельвинах;
double M = 1.2873;
double X_H20 = 3;

double Coef_Nucl_mult = Lx * Lx * t_end_yrs * 0.6e+1;
double Coef_delta_C = 3.0;    //    чем больше, тем более плоский график, чем меньше, тем быстрее наступит пик, начало остаётся примерно такое же
double Coef_N = 2.9; //    сдвиг пика влево и вверх при ывеличении, а также уменьшение значений меньше 0
double Coef_nucl_c = 0.06e-2; // уменьшение - снижает значимость разницы концентраций
double Coef_nucl_T = 0.4e+4;  // увеличение - опускает всю кривую, а именно начальную точку, отдаляет пик от начала
//чтобы увеличить пик - нужно увеличить N и delta_C ?
*/


//HRT-C +
/*
* 
* 
* l = 0.15
double Coef_Nucl_mult = Lx * Lx * t_end_yrs * 0.6e+1;
double Coef_delta_C = 4.7;    //    чем больше, тем более плоский график, чем меньше, тем быстрее наступит пик, начало остаётся примерно такое же
double Coef_N = 5.53; //    сдвиг пика влево и вверх при ывеличении, а также уменьшение значений меньше 0
double Coef_nucl_c = 0.06e-2; // уменьшение - снижает значимость разницы концентраций
double Coef_nucl_T = 0.05e+5;  // увеличение - опускает всю кривую, а именно начальную точку, отдаляет пик от начала
//чтобы увеличить пик - нужно увеличить N и delta_C ?
*/