#pragma once

#include <iostream>
#include <time.h>
#include <algorithm>

// gsl
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_qrng.h>
#include <gsl/gsl_randist.h>

// opencv
#include <core.hpp>
#include <highgui.hpp>
#include <opencv.hpp>
#include <imgproc.hpp>
#include <objdetect.hpp>
#include <imgproc/types_c.h>


#define NUM_PARTICLE 50 // 粒子数

using namespace std;
using namespace cv;

/*********************结构体************************/
// 粒子结构体
typedef struct particle {
	double x;				// 当前x坐标
	double y;				// 当前y坐标
	double scale;			// 窗口比例系数
	double xPre;			// x坐标预测位置
	double yPre;			// y坐标预测位置
	double scalePre;		// 窗口预测比例系数
	double xOri;			// 原始x坐标
	double yOri;			// 原始y坐标
	int width;				// 原始区域宽度
	int height;				// 原始区域高度
	MatND hist;				// 粒子区域的特征直方图
	double weight;			// 该粒子的权重
} PARTICLE;

/************************粒子状态转移（新位置生成预测）***********************/
/* standard deviations for gaussian sampling in transition model */
#define TRANS_X_STD 1.0
#define TRANS_Y_STD 0.5
#define TRANS_S_STD 0.001
/* autoregressive dynamics parameters for transition model */
#define A1  2.0//2.0
#define A2  -1.0//-1.0
#define B0  1.0000

/************************鼠标回调部分*************************/
Rect roiRect;//选取矩形区
Point startPoint;//起点
Point endPoint;//终点
Mat hsv_roiImage;
Mat current_frame;
Mat roiImage;
bool downFlag = false;// 按下标志位
bool upFlag = false;// 弹起标志位
bool getTargetFlag = false;
void MouseEvent(int, int, int, int, void*);

// 提取感兴趣Mat元
Mat regionExtraction(int, int, int, int);

/***********************直方图参数********************************/
// 直方图
int hbins = 10, sbins = 10, vbin = 20;  //180 256 10
int histSize[] = { hbins, sbins };//vbin
//h的范围
float hranges[] = { 0, 180 };
//s的范围
float sranges[] = { 0, 256 };
float vranges[] = { 0, 256 };

// 比较hsv模型的色调和饱和度两个通道
const float* ranges[] = { hranges, sranges };

// 比较直方图的0-th 和 1-st 通道
int channels[] = { 0, 1 };

class AutoTrace
{
public:
	// 粒子初始化
	void particle_init(particle*, int, MatND);
	// 粒子状态转移（新位置生成预测）
	particle transition(particle p, int w, int h, gsl_rng* rng);
	// 粒子权重归一化
	void normalize_weights(particle*, int);
	// 粒子重采样
	void resample(particle*, particle*, int);
	// 二分法求数组中大于给定值的最小值索引
	static int get_min_index(double*, int, double);
	// 粒子权重排序
	static int particle_cmp(const void*, const void*);
public:
	// 创建粒子数组
	int num_particles = NUM_PARTICLE; // 粒子数
	PARTICLE particles[NUM_PARTICLE];
	PARTICLE new_particles[NUM_PARTICLE];
	// 视频素材
	string video_name;
};