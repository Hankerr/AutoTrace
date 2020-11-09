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

// 素材修改，替换material.avi
string video_file_name = "material.avi";

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

#define TRANS_X_STD 1.0
#define TRANS_Y_STD 0.5
#define TRANS_S_STD 0.001
/* autoregressive dynamics parameters for transition model */
#define A1  2.0//2.0
#define A2  -1.0//-1.0
#define B0  1.0000

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

// 直方图参数
// 直方图
int hbins = 10, sbins = 10, vbin = 20;
int histSize[] = { hbins, sbins };
//h的范围
float hranges[] = { 0, 180 };
//s的范围
float sranges[] = { 0, 256 };
float vranges[] = { 0, 256 };

// 比较hsv模型的色调和饱和度两个通道
const float* ranges[] = { hranges, sranges };

// 比较直方图的0-th 和 1-st 通道
int channels[] = { 0, 1 };

bool readFrameFlag = false;
