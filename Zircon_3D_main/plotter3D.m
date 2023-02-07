%clc, clear
FLAG_numbers = 1;

HRT_C_raspr =[0.11088,	0.158996,	0.239538,	0.285264,	0.369737,	0.408577,	0.451856,	0.49503,	0.54475,	0.581224,	0.617807,	0.704456,	0.754303,	0.788375,	0.833986,	0.876848,	0.920427,	1.00239;...
             -3.25333,	-1.992,	-0.833333,	-0.144,	0.00266667,	-0.305333,	-0.334667,	-0.628,	-0.789333,	-1.61067,	-2.15333,	-1.97733,	-1.816,	-3.23867,	-2.84267,	-3.928,	-3.19467,	-3.928];
MFT_1_raspr       = [0.0525426,	0.0943261,	0.137887,	0.181224,	0.222326,	0.270061,	0.315608,	0.358926,	0.404387,	0.438984,	0.48228,	0.529824,	0.568791,	0.614136,	0.659544,	0.696294,	0.741755,	0.7871,	0.826084,	0.871377,	0.91032,	0.957835,	1.00084;...
              -2.98933,	-1.31733,	-0.628,	-0.510667,	-0.569333,	-0.276,	-0.0413333,	0.032,	0.0466667,	-0.0413333,	-0.0266667,	-0.217333,	-0.202667,	-0.481333,	-0.598667,	-0.716,	-0.701333,	-0.98,	-0.921333,	-1.332,	-1.376,	-1.64,	-2.35867];

HRT_A_raspr = [21.7068645640070,32.8385899814470,43.4137291280150,54.5454545454540,74.5825602968460,89.6103896103900,95.1762523191090,106.307977736550,120.222634508350,128.571428571430,136.363636363640,150.834879406310,161.966604823750,170.871985157700,184.230055658630,204.823747680890,217.068645640070;8.91585552790700,9.76402383443680,9.80327503685660,9.43817320120640,8.80879400446090,8.52001395129360,8.30586745684920,8.24408315674390,8.01835357150180,7.81719702098850,8.00775527789750,7.82001072725510,7.55601473678660,7.19063153050970,6.80053460419070,6.56301090018080,6.19804974984390];

HRT_C_raspr_try = [24.4821,	34.0386,	42.7092,	55.0554,	64.5076,	76.7087,	83.9091,	93.1785,	106.097,	114.661,	123.962,	136.018,	145.262,	155.333,	164.663,	175.449,	183.821,	198.242,	205.225,	215.991,	234.651;
                   6.49557,	7.7572,	8.08137,	8.89078,	9.57059,	9.5719,	9.70197,	9.36358,	9.36497,	9.09114,	8.93053,	8.12375,	7.63989,	7.77027,	7.77128,	7.88557,	6.54506,	6.91833,	5.83625,	5.83741,	5.83942];
LV_13 = [6.50759,	14.9675,	26.6811,	45.5531,	55.9653,	65.7267,	78.0911,	86.551,	    96.3124,	101.518,	107.375,	115.835,	125.597,	136.659,	156.182,	165.944,	185.466,	195.228;
                   8.47928,	10.0418,	9.93109,	8.37062,	8.03671,	7.36807,	7.48055,	6.76718,	6.43322,	5.5188,	    5.13993,	4.8505,	    4.20416,	3.84799, 	 4.1395,    3.24773,	3.24918,	2.73673];
 
LV_3  = [     15,	  23.4783,	34.5652,	43.0435,	54.1304,	62.6087,	75,	83.4783,	94.5652,	101.739,	114.783,	123.913,	133.696,	143.478,	155.217,	165,	174.783,	184.565,	194.348,	205.435;...
         3.95935,	6.49593,	6.23577,	7.34146,	7.66667,	8.04065,	7.87805,	8.12195,	8.07317,	7.79675,	7.53659,	7.34146,	7.09756,	6.85366,	6.73984,	6.56098,	5.92683,	5.42276,	4.70732,	4.69106];

TM_15 = [13.0435,	23.4783,	35.2174,	44.3478,	56.087,	63.913,	74.3478,	84.7826,	93.913,	104.348,	116.087,	122.609,	134.348,	143.478,	155.217,	164.348,	174.783,	185.217,	203.478,	284.348,	293.478;...
        6.41463,	7.13008,	7.35772,	8.82114,	9.47154,	9.47154,	8.9187,	8.85366,	7.84553,	7.81301,	7.71545,	7.09756,	6.6748,	6.5122,	6.70732,	6.41463,	5.27642,	5.69919,	4.23577,	3.19512,	3.09756];
    
YL_4 = [15.6398,	24.8815,	54.7393,	76.0664,	83.8863,	90.9953,	100.948,	113.033,	125.118,	135.782,	145.735,	157.82,	171.327,	177.014,	196.209,	228.199,	274.408;...
        4.35603,	6.11112,	7.1648,	7.58525,	7.5844,	7.43729,	6.98093,	6.65441,	6.7344,	6.48935,	5.59396,	5.52761,	5.15216,	4.68,	4.22263,	4.0403,	3.07594];

    
LCT_3a = [23.4597,	34.1232,	42.654,	65.4028,	73.2227,	81.7536,	92.4171,	105.213,	113.744,	122.986,	132.938,	144.313,	154.265,	163.507,	175.592,	184.834,	193.365,	204.028,	214.692,	245.261;...
          6.3877,	5.88248,	5.93034,	8.17177,	8.36605,	8.98301,	8.36397,	8.4764,	8.03645,	7.58016,	7.00998,    6.84614,	5.88572,	5.51073,	4.77771,	5.42712,	5.45871,	4.80715,	4.74095,	3.59943];
      
%  YL_4 = [17.9479	18.9975	19.7316	20.5443	21.436	22.17	22.9827	23.7164	24.3714	25.2631	26.6281	28.3874	30.6197	33.4037	36.5035	39.6824	42.7823	46.0399	49.3764	52.792	56.1285	59.7007	63.745	68.1839	72.8587	77.9231	83.1359	88.1026	92.1151	95.0178	97.1287	99.161	101.826	105.125	109.057	113.709	119.001	124.292	129.261	133.434	136.496	138.448	139.926	141.326	142.646	143.966	145.761	148.825	153.396	158.839	164.203	16.8192	16.2431	15.0358	202.521	207.965	213.489	219.093	224.696	229.821	234.231	238.481	242.651	245.95	247.192	197.157	192.668	188.656	184.723	180.712	177.254	175.219	173.976	172.496	170.384	167.481	164.499	162.309	271.448	274.272	267.436	263.424	259.253	255.242	250.991	245.787	276.652;...
 %     4.76421	4.8926	5.02276	5.15112	5.2795	5.40966	5.53802	5.66637	5.79653	5.9249	6.04971	6.17094	6.28681	6.39552	6.50425	6.613	6.72174	6.82868	6.93564	7.04261	7.14956	7.24932	7.33828	7.42548	7.50729	7.55843	7.55538	7.50172	7.41182	7.30012	7.17749	7.05667	6.94313	6.83509	6.73976	6.69149	6.68845	6.6836	6.64258	6.56354	6.45909	6.33644	6.20833	6.08201	5.95569	5.82756	5.7049	5.60947	5.55035	5.51119	5.47022	4.63582	4.50567	4.37907	4.18621	4.14886	4.12236	4.09767	4.06576	4.01392	3.93852	3.85226	3.76057	3.65794	3.53161	4.22537	4.30077	4.39247	4.4878	4.58312	4.68031	4.78668	4.90217	5.02125	5.13846	5.24835	5.35824	5.47905	3.12774	3.01422	3.21944	3.30934	3.39922	3.49274	3.57538	3.62541	3.0741];

    
HRT_A_raspr(1,:) = HRT_A_raspr(1,:)/max(HRT_A_raspr(1,:));
HRT_A_raspr(2,:) = log(exp(HRT_A_raspr(2,:))/exp(max(HRT_A_raspr(2,:))));       
               
LV_13(1,:) = LV_13(1,:)/max(LV_13(1,:));
LV_13(2,:) = log(exp(LV_13(2,:))/exp(max(LV_13(2,:))));              
               
HRT_C_raspr_try(1,:) = HRT_C_raspr_try(1,:)/max(HRT_C_raspr_try(1,:));
HRT_C_raspr_try(2,:) = log(exp(HRT_C_raspr_try(2,:))/exp(max(HRT_C_raspr_try(2,:))));

LV_3(1,:) = LV_3(1,:)/max(LV_3(1,:));
LV_3(2,:) = log(exp(LV_3(2,:))/exp(max(LV_3(2,:)))); 

TM_15(1,:) = TM_15(1,:)/max(TM_15(1,:));
TM_15(2,:) = log(exp(TM_15(2,:))/exp(max(TM_15(2,:)))); 

YL_4(1,:) = YL_4(1,:)/max(YL_4(1,:));
YL_4(2,:) = log(exp(YL_4(2,:))/exp(max(YL_4(2,:)))); 

LCT_3a(1,:) = LCT_3a(1,:)/max(LCT_3a(1,:));
LCT_3a(2,:) = log(exp(LCT_3a(2,:))/exp(max(LCT_3a(2,:)))); 


Graph = figure('Units','normalized','Position',[.10,.10,.8,.8]);
ha = axes('Units','normalized','Position',[.05,.48,.40,0.47]);
      
set(Graph,'Position',[.10,.10,.8,.8]);
box on
axes_raspr = axes('Units','normalized','Position',[0.05,0.15,0.40,0.23]); box on
axes_vel   = axes('Units','normalized','Position',[0.5,0.65,0.40,0.30]); box on
axes_tempr = axes('Units','normalized','Position',[0.5,0.4,0.40,0.15]); box on

      
xlabel(axes_vel,'Time (years)');
ylabel(axes_vel,'Zr radius');

xlabel(axes_tempr,'Time (years)');
ylabel(axes_tempr,'Growth Rate, cm.s^{-1}');



set(Graph,'Name','DIFFUSOR');
movegui(Graph,'center');
set(Graph,'NumberTitle','off');

set(Graph,'CurrentAxes',ha);


nx = 64;
L_size = 0.005;
Drob                = 20;

ny = nx;
nt = nx*nx*nx;
%nx = 3;
%ny = 4;
qq = [1 1;2 2;3 3];
S = 2*5*1.e-6;
%{
if(S/L_size/2<1/(nx-1))
    S = 2*1/(nx-1)*L_size;
end
%}
t_end_yrs = 10000;
for i = 999:999
fileID = fopen([num2str(i),'.dat'],'r');
formatSpec = '%f';
sizeA = [nx ny];
size_time = [2*nt 1];
%A = fscanf(fileID,formatSpec,sizeA)';
%A = flip(A,1);
Lx = 1;
Ly = Lx;

fileID_1 = fopen(['1.dat'],'r');
fileID_2 = fopen(['2.dat'],'r');
fileID_3 = fopen(['3.dat'],'r');

fileID_10004 = fopen(['10004.dat'],'r');
fileID_10005 = fopen(['10005.dat'],'r');
fileID_10006 = fopen(['10006.dat'],'r');
fileID_10007 = fopen(['10007.dat'],'r');
fileID_10008 = fopen(['10008.dat'],'r');
fileID_10009 = fopen(['10009.dat'],'r');


fileID_8 = fopen(['8.dat'],'r');

time   = fscanf(fileID_1,formatSpec,size_time)';
V_nucl = fscanf(fileID_2,formatSpec,size_time)';
Tempr  = fscanf(fileID_3,formatSpec,size_time)';

S_left  = fscanf(fileID_10004,formatSpec,size_time)';
S_right = fscanf(fileID_10005,formatSpec,size_time)';
S_top   = fscanf(fileID_10006,formatSpec,size_time)';
S_bot   = fscanf(fileID_10007,formatSpec,size_time)';
S_back = fscanf(fileID_10008,formatSpec,size_time)';
S_forth  = fscanf(fileID_10009,formatSpec,size_time)';


Cryst_age  = fscanf(fileID_8,formatSpec,size_time)';

n = size(S_left,2);



S_left(1:n)  = S_left(1:n)*L_size;
S_right(1:n) = S_right(1:n)*L_size;
S_top(1:n)   = S_top(1:n)*L_size;
S_bot(1:n)   = S_bot(1:n)*L_size;
S_back(1:n)  = S_back(1:n)*L_size;
S_forth(1:n) = S_forth(1:n)*L_size;




Crystal_size_x      = linspace(0,1,Drob);
Crystal_size_y      = zeros(1,Drob);
Crystal_size        = zeros(1,size(S_left,2));
qq = S_right(1:n)-S_left(1:n) - S ;
qq_vert = S_top(1)-S_bot(1) - S ;

Crystal_size(1:n) = ((S_right(1:n)-S_left(1:n)).^2 + (S_top(1:n)-S_bot(1:n)).^2 + (S_forth(1:n)-S_back(1:n)).^2).^(1/2);

%Crystal_size(1:n) = Crystal_size(1:n) - 2^(1/2)*S;
    Crystal_size_max = Crystal_size(1);
    first_cryst      = 1;
   
    
    for m=first_cryst:n
        if(Crystal_size_max<Crystal_size(m))
            Crystal_size_max = Crystal_size(m);
        end
    end
    
    qsc=Crystal_size/Crystal_size_max;
    
    
    
    %Crystal_size_max=9.208604625814420e-06;
    
    for ii=first_cryst:n
        r = 1;
     while Crystal_size_x(r)<(Crystal_size(ii)/Crystal_size_max)
        r = r+1;
     end
        Crystal_size_y(r)  = Crystal_size_y(r)+1;
    end
    
        Number_crystals_max = 1;
    for j=1:Drob
        if (Crystal_size_y(j)>=Number_crystals_max)
            Number_crystals_max = Crystal_size_y(j);
        end
    end


c = colorbar;
Lz = Lx;
dx = Lx/(nx-1);
dy = Ly/(ny-1);
dz = Ly/(ny-1);
%{
[X,Y] = ndgrid(0:dx:Lx,0:dy:Ly);
X     = X';
Y     = flip(Y',1);
%}
set(Graph,'CurrentAxes',ha);

x = -2:.2:2;
y = -2:.25:2;
z = -2:.16:2;

xx = 0:dx:Lx;
yy = 0:dy:Ly;
zz = 0:dz:Lz;

[X,Y,Z] = meshgrid(0:dx:Lx,0:dy:Ly,0:dz:Lz);
v = xx.*exp(-xx.^2-yy.^2-zz.^2);
%(int)(0.5 * (NX - 1)) / (double)(NX - 1)
bb = 0;
xslice = [ fix(0.5 * (nx - 1))/(nx - 1)];       % location of y-z planes
yslice = [ .5];       % location of x-z plane
zslice = [0, .5];         % location of x-y planes


fileID_10 = fopen(['999.dat'],'r');
formatSpec = '%f';
sizeAA = [nx*nx*nx];

AA = fscanf(fileID,formatSpec,sizeAA);
AA = reshape(AA, nx, ny, []);

for ii =1:nx
    AA(:,:,ii) = AA(:,:,ii)';
end

%AA = flip(AA,1);
%AA = flip(AA,2);
%AA = flip(AA,3);



s = slice(X,Y,Z,AA,xslice,yslice,zslice)
xlabel('x')
ylabel('y')
zlabel('z')
alpha 0.8
axis image;c = colorbar;drawnow
%c.Limits = [160 200]

colormap(jet);
light;
shading interp;

%{
pcolor(X,Y,A);

axis image;c = colorbar;drawnow
%s.LineWidth =0;
colormap(jet);
xlabel('x');
ylabel('y');
%grid off;
light;
shading interp;
%}


if (FLAG_numbers ==1)
 str_N_cryst           = {'N cryst=',num2str(n)};
 
      text(0,-0.85,str_N_cryst);


      for j=1:n
          text((S_right(j) + S_left(j))/2/L_size,(S_top(j) + S_bot(j))/2/L_size,(S_forth(j) + S_back(j))/2/L_size,num2str(j), 'Color', 'black','FontSize', 10);
      end
end

%axis off;
%c.Limits = [160 200]

a=gca;

set(a,'xlim',[0 1]);
set(a,'ylim',[0 1]);
set(a,'zlim',[0 1]);

% Generate constants for use in uicontrol initialization
pos=get(a,'position');
Newpos=[pos(1) pos(2)-0.1 pos(3) 0.05];
xxx_slide = 0.5;
yyy_slide = 0.5;
zzz_slide = 0.5;
xmax=max(x);
%S=['set(s,''xlim'',get(gcbo,''value'')+[0 ' num2str(dx) '])'];
HEHE = view();
%S_x = ['xxx_slide = get(gcbo,''value'');xslice = [ xxx_slide ];yslice = [ yyy_slide];zslice = [0, zzz_slide]; slice(X,Y,Z,AA,xslice,yslice,zslice);axis image;c = colorbar;drawnow;c.Limits = [160 200];colormap(jet);light;shading interp;']
S_x = ['plotter_3D(get(gcbo,''value''), yyy_slide, zzz_slide,X,Y,Z,AA,n,S_right,S_left,S_top,S_bot,S_forth,S_back,L_size,FLAG_numbers);']


%S_y = ['yyy_slide = get(gcbo,''value'');xslice = [ xxx_slide ];yslice = [ yyy_slide];zslice = [0, zzz_slide]; slice(X,Y,Z,AA,xslice,yslice,zslice);axis image;c = colorbar;drawnow;c.Limits = [160 200];colormap(jet);light;shading interp;']
S_y = ['plotter_3D(xxx_slide, get(gcbo,''value''), zzz_slide,X,Y,Z,AA,n,S_right,S_left,S_top,S_bot,S_forth,S_back,L_size,FLAG_numbers);']

%S_z = ['zzz_slide = get(gcbo,''value'');xslice = [ xxx_slide ];yslice = [ yyy_slide];zslice = [0, zzz_slide]; slice(X,Y,Z,AA,xslice,yslice,zslice);axis image;c = colorbar;drawnow;c.Limits = [160 200];colormap(jet);light;shading interp;']
S_z = ['plotter_3D(xxx_slide, yyy_slide, get(gcbo,''value''),X,Y,Z,AA,n,S_right,S_left,S_top,S_bot,S_forth,S_back,L_size,FLAG_numbers);']

pos=get(a,'position');
Newpos_x=[0.5+pos(1) pos(2)-0.2 pos(3) 0.05];
Newpos_y=[0.5+pos(1) pos(2)-0.3 pos(3) 0.05];
Newpos_z=[0.5+pos(1) pos(2)-0.4 pos(3) 0.05];

% Creating Uicontrol
 h_x=uicontrol('style','slider','units','normalized','position',Newpos_x,'callback',S_x,'min',0,'max',1);
 h_y=uicontrol('style','slider','units','normalized','position',Newpos_y,'callback',S_y,'min',0,'max',1);
 h_z=uicontrol('style','slider','units','normalized','position',Newpos_z,'callback',S_z,'min',0,'max',1);
 
 
%S=['set(gca,''xlim'',get(gcbo,''value'')+[0 ' num2str(dx) '])'];

set(Graph,'CurrentAxes',axes_vel);
plot(time*t_end_yrs,log(V_nucl));      
set(axes_vel,'xlim',[0 t_end_yrs]);
set(axes_vel,'ylim',[-3 15]);
      xlabel(axes_vel,'Время (лет)')
      ylabel(axes_vel,'ln(J)') 
      
      set(Graph,'CurrentAxes',axes_tempr)
      plot(time*t_end_yrs,Tempr-273);
      set(axes_tempr,'xlim',[0 t_end_yrs]);
      set(axes_tempr,'ylim',[min(Tempr(Tempr>0))-273-0.000000001 max(Tempr(Tempr>0))-273]);
      xlabel(axes_tempr,'Время (лет)')
      ylabel(axes_tempr,'Temperature T, ^oC')
      
      set(Graph,'CurrentAxes',axes_raspr);
      %set(axes_raspr, 'YScale', 'log')
      set(axes_raspr,'xlim',[0 1]);
     set(axes_raspr,'ylim',[-8 1]);
      
      plot(HRT_C_raspr_try(1,:),HRT_C_raspr_try(2,:),Crystal_size_x,log(Crystal_size_y/Number_crystals_max),'o');
      legend('HRT-C','Численное решение');

      qq_vert/5.552530092907038e-08

      %max(max(A))   
      %min(min(A))
      [asd,zxc] = min(min(min(AA)))

      [asd,zxc(1,1,1)] = max(AA,[],'all')
      max(Crystal_size)
      %AA(1,1,1:10)
      asd = max(AA,[],'all')
      i
end


set(Graph,'CurrentAxes',ha);
 %   0.9385 - 1 - 2048 - 10000 yrs
%max(max(A))
%min(min(A))
max(max(qq/1.336485118872319e-06))

sum(Cryst_age)/n

%{
const int NX      = 2048;
const int NY      = NX;            
int N_cryst_max   = 1000;
int Nt = 500, MAX_ITER = 1;
double T_start    = (double)897 + 273;     // начальная температура в кельвинах;
double T_end      = (double)888.8804 + 273;     //конечная температура в кельвинах;
double Lx         = 0.1, D = 0.1;
double dx         = 1.0 / (NX - 1);
double dy         = dx;
double t_end_yrs  = 10000;
double time_coef  = (T_start - T_end);
double t_end      = t_end_yrs * 365 * 24 * 60 * 60.0; // окончание по времени;


int n_img				= 1;
double V_nucl           = 0;
double max_delta		= 0;
double Numb_cryst_begin = 0;

double a = 3.e-5; // меньше - больше ограничения, чем меньше, тем медленнее перешагивает

int gap = NX/20;

double eps = 0.01;

int N_dt_razb = 10;
int FLAG_new_cryst = 1; //help flag
int FLAGGG_counter = 0;;
int FLAG_k_top_k_bot = 0;

int Graph_num = 10;
int SIZE_OF_BLOCK = 32;
double UNO            = 1;
double C_depth_new    = 5;

int FLAG_create_cryst = 1; 

int FLAG_all_graphs = 0;

int FLAG_1d_test = 0;
int FLAG_new_progonka = 0;

double C_cryst = 490000;


double M = 1.2607;
double C_sat = C_cryst / (exp(12900 / T_start - 0.85 * (M - 1) - 3.80));
double X_H20 = 3;
double init_value = C_sat;



double Coef_Nucl_mult   = Lx * Lx * t_end_yrs * 0.5e+1;
double Coef_delta_C     = 8.7;    //    чем больше, тем более плоский график, чем меньше, тем быстрее наступит пик, начало остаётся примерно такое же
double Coef_N           = 7.50; //    сдвиг пика влево и вверх при ывеличении, а также уменьшение значений меньше 0
double Coef_nucl_c      = 0.06e-2; // уменьшение - снижает значимость разницы концентраций
double Coef_nucl_T      = 0.05e+5;  // увеличение - опускает всю кривую, а именно начальную точку, отдаляет пик от начала
//чтобы увеличить пик - нужно увеличить N и delta_C
%}

function plotter_3D(xxx_slide_in, yyy_slide_in, zzz_slide_in)
        xslice = [ xxx_slide_in ];
        yslice = [ yyy_slide_in];
        zslice = [0, zzz_slide_in]; 
        slice(X,Y,Z,AA,xslice,yslice,zslice);
        axis image;
        c = colorbar;
        drawnow;
        %c.Limits = [160 200];
        colormap(jet);
        light;
        shading interp;


      str_N_cryst           = {'N cryst=',num2str(n)};
 
      text(0,-0.85,str_N_cryst);


      for j=1:n
          text((S_right(j) + S_right(j))/2/L_size,(S_top(j) + S_bot(j))/2/L_size,(S_forth(j) + S_back(j))/2/L_size,num2str(j), 'Color', 'black','FontSize', 10);
      end

end