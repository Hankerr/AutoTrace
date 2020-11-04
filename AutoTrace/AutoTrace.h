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


#define NUM_PARTICLE 50 // ������

using namespace std;
using namespace cv;

/*********************�ṹ��************************/
// ���ӽṹ��
typedef struct particle {
	double x;				// ��ǰx����
	double y;				// ��ǰy����
	double scale;			// ���ڱ���ϵ��
	double xPre;			// x����Ԥ��λ��
	double yPre;			// y����Ԥ��λ��
	double scalePre;		// ����Ԥ�����ϵ��
	double xOri;			// ԭʼx����
	double yOri;			// ԭʼy����
	int width;				// ԭʼ������
	int height;				// ԭʼ����߶�
	MatND hist;				// �������������ֱ��ͼ
	double weight;			// �����ӵ�Ȩ��
} PARTICLE;

/************************����״̬ת�ƣ���λ������Ԥ�⣩***********************/
/* standard deviations for gaussian sampling in transition model */
#define TRANS_X_STD 1.0
#define TRANS_Y_STD 0.5
#define TRANS_S_STD 0.001
/* autoregressive dynamics parameters for transition model */
#define A1  2.0//2.0
#define A2  -1.0//-1.0
#define B0  1.0000

/************************���ص�����*************************/
Rect roiRect;//ѡȡ������
Point startPoint;//���
Point endPoint;//�յ�
Mat hsv_roiImage;
Mat current_frame;
Mat roiImage;
bool downFlag = false;// ���±�־λ
bool upFlag = false;// �����־λ
bool getTargetFlag = false;
void MouseEvent(int, int, int, int, void*);

// ��ȡ����ȤMatԪ
Mat regionExtraction(int, int, int, int);

/***********************ֱ��ͼ����********************************/
// ֱ��ͼ
int hbins = 10, sbins = 10, vbin = 20;  //180 256 10
int histSize[] = { hbins, sbins };//vbin
//h�ķ�Χ
float hranges[] = { 0, 180 };
//s�ķ�Χ
float sranges[] = { 0, 256 };
float vranges[] = { 0, 256 };

// �Ƚ�hsvģ�͵�ɫ���ͱ��Ͷ�����ͨ��
const float* ranges[] = { hranges, sranges };

// �Ƚ�ֱ��ͼ��0-th �� 1-st ͨ��
int channels[] = { 0, 1 };

class AutoTrace
{
public:
	// ���ӳ�ʼ��
	void particle_init(particle*, int, MatND);
	// ����״̬ת�ƣ���λ������Ԥ�⣩
	particle transition(particle p, int w, int h, gsl_rng* rng);
	// ����Ȩ�ع�һ��
	void normalize_weights(particle*, int);
	// �����ز���
	void resample(particle*, particle*, int);
	// ���ַ��������д��ڸ���ֵ����Сֵ����
	static int get_min_index(double*, int, double);
	// ����Ȩ������
	static int particle_cmp(const void*, const void*);
public:
	// ������������
	int num_particles = NUM_PARTICLE; // ������
	PARTICLE particles[NUM_PARTICLE];
	PARTICLE new_particles[NUM_PARTICLE];
	// ��Ƶ�ز�
	string video_name;
};