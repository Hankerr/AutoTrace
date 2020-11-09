#include "AutoTrace.h"

int get_min_index(double*, int, double);
Mat regionExtraction(int, int, int, int);
void MouseEvent(int, int, int, int, void*);
void particle_init(particle*, int, MatND);
particle transition(particle, int, int, gsl_rng*);
void normalize_weights(particle*, int);
int particle_cmp(const void*, const void*);
void resample(particle*, particle*, int);
int get_min_index(double*, int, double);

int main()
{
	Mat frame, hsv_frame;
	vector<Mat> frames;
	// 目标的直方图
	MatND hist;
	VideoCapture capture(video_file_name);	// 视频文件video_file_name

	// 粒子数
	int num_particles = NUM_PARTICLE;
	PARTICLE particles[NUM_PARTICLE];
	PARTICLE new_particles[NUM_PARTICLE];
	PARTICLE * pParticles;
	pParticles = particles;;
	//随机数生成器
	gsl_rng* rng;
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rng, time(NULL));
	float s;
	int j;
	// 判断视频是否打开
	if (!capture.isOpened())
	{
		cout << "Capture open fail" << endl;
		system("pause");
		return -1;
	}
	// 读取一帧
	while (1) {
		capture >> frame;
		if (frame.empty()) {
			cout << "Read frame finish" << endl;
			readFrameFlag = true;
			break;
		}
		// 创建窗口
		namedWindow("frame", WINDOW_NORMAL);
		// 复制一个原始帧，给框定目标回调函数用
		current_frame = frame.clone();
		setMouseCallback("frame", MouseEvent, "frame");
		frames.push_back(frame.clone());
		imshow("frame", frame);
		cv::waitKey(40);
		if (getTargetFlag == true) {
			// 目标区域转换到hsv空间
			cvtColor(roiImage, hsv_roiImage, COLOR_BGR2HSV);
			// 计算目标区域的直方图
			calcHist(&hsv_roiImage, 1, channels, Mat(), hist, 2, histSize, ranges);
			// 归一化L2
			normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
			// 粒子初始化
			particle_init(particles, num_particles, hist);
		}
		else {
			cout << "未抓取目标，请抓取目标..." << endl;
			continue;
		}
		while (1) {
			capture >> frame;
			if (frame.empty()) {
				cout << "finish" << endl;
				break;
			}
			current_frame = frame.clone();
			frames.push_back(frame.clone());

			// 对每个粒子的操作：
			for (j = 0; j < num_particles; j++) {
				// 这里利用高斯分布的随机数来生成每个粒子下一次的位置以及范围
				particles[j] = transition(particles[j], frame.cols, frame.rows, rng);
				s = particles[j].scale;
				// 根据新生成的粒子信息截取对应frame上的区域
				Rect imgParticleRect = Rect(std::max(0, std::min(cvRound(particles[j].x - 0.5*particles[j].width), cvRound(frame.cols - particles[j].width*s))),
					std::max(0, std::min(cvRound(particles[j].y - 0.5*particles[j].height), cvRound(frame.rows - particles[j].height*s))),
					cvRound(particles[j].width*s),
					cvRound(particles[j].height*s));

				Mat imgParticle = current_frame(imgParticleRect).clone();
				// 上述区域转换到hsv空间
				cvtColor(imgParticle, imgParticle, COLOR_BGR2HSV);
				// 计算区域的直方图
				calcHist(&imgParticle, 1, channels, Mat(), particles[j].hist, 2, histSize, ranges);
				// 归一化L2
				// 直方图归一化到（0，1）
				normalize(particles[j].hist, particles[j].hist, 0, 1, NORM_MINMAX, -1, Mat());
				// 画出蓝色的粒子框
				rectangle(frame, imgParticleRect, Scalar(255, 0, 0), 1, 8);
				// 调试打开
				// imshow("particle", imgParticle);
				// 比较目标的直方图和上述计算的区域直方图,更新particle权重
				particles[j].weight = exp(-100 * (compareHist(hist, particles[j].hist, CV_COMP_BHATTACHARYYA)));
			}
			// 归一化权重 
			normalize_weights(particles, num_particles);

			// 重采样
			int np, k = 0;
			// 将粒子按权重从高到低排序
			qsort(particles, num_particles, sizeof(particle), &particle_cmp);
			// 重采样
			resample(particles, new_particles, num_particles);
			// 重排序
			qsort(particles, num_particles, sizeof(particle), &particle_cmp);
			// 取权重最高的作为目标，标准做法：按加权平均来计算目标位置
			s = particles[0].scale;
			Rect rectTrackingTemp = Rect(std::max(0, std::min(cvRound(particles[0].x - 0.5*particles[0].width), cvRound(frame.cols - particles[0].width*s))),
				std::max(0, std::min(cvRound(particles[0].y - 0.5*particles[0].height), cvRound(frame.rows - particles[0].height*s))),
				cvRound(particles[0].width*s),
				cvRound(particles[0].height*s));
			rectangle(frame, rectTrackingTemp, Scalar(0, 0, 255), 1, 8, 0);
			cout << "计算得到区域(x1,y1):" << rectTrackingTemp.br().x << "," << rectTrackingTemp.br().y << 
				"(x2,y2):" << rectTrackingTemp.br().x+ rectTrackingTemp.width << "," << rectTrackingTemp.br().x + rectTrackingTemp.height << endl;
			imshow("frame", frame);
			cv::waitKey(40);
		}
		if (readFrameFlag)
		{
			return -1;
		}
	}
	cout << "Application finish" << endl;
	system("pause");
	return 0;
}

// 选中目标源显示
Mat regionExtraction(int xRoi, int yRoi, int widthRoi, int heightRoi)
{
	//创建与原图像同大小的Mat
	Mat roiImage;
	//提取感兴趣区域
	roiImage = current_frame(Rect(xRoi, yRoi, widthRoi, heightRoi)).clone();
	imshow("选中素材", roiImage);
	return roiImage;
}

// 鼠标事件回调方法
void MouseEvent(int event, int x, int y, int flags, void* win_name)
{
	//左键按下，取当前位置
	if (event == EVENT_LBUTTONDOWN) {
		downFlag = true;
		getTargetFlag = false;
		startPoint.x = x;
		startPoint.y = y;
	}
	//弹起，取当前位置作为终点
	if (event == EVENT_LBUTTONUP) {
		upFlag = true;
		endPoint.x = x;
		endPoint.y = y;
		//终点最值限定
		if (endPoint.x > current_frame.cols)endPoint.x = current_frame.cols;
		if (endPoint.y > current_frame.cols)endPoint.y = current_frame.rows;
	}
	//显示区域
	if (downFlag == true && upFlag == false) {
		Point tempPoint;
		tempPoint.x = x;
		tempPoint.y = y;
		// 取原图像复制
		Mat tempImage = current_frame.clone();
		//用矩形标记
		rectangle(tempImage, startPoint, tempPoint, Scalar(0, 0, 255), 2, 3, 0);
		//imshow((char*)data, tempImage);
		imshow((char*)win_name, tempImage);
	}
	//按下选取完并弹起后
	if (downFlag == true && upFlag == true) {
		//起点和终点不相同时，才提取区域
		if (startPoint.x != endPoint.x&&startPoint.y != endPoint.y) {
			startPoint.x = min(startPoint.x, endPoint.x);
			startPoint.y = min(startPoint.y, endPoint.y);
			roiRect = Rect(startPoint.x, startPoint.y, endPoint.x - startPoint.x, endPoint.y - startPoint.y);
			roiImage = regionExtraction(startPoint.x, startPoint.y,
				abs(startPoint.x - endPoint.x),
				abs(startPoint.y - endPoint.y));
		}
		downFlag = false;
		upFlag = false;
		getTargetFlag = true;
		cout << "抓取目标区域(x1,y1):" << startPoint.x <<"," <<startPoint.y << "(x2,y2):" << endPoint.x << "," << endPoint.y << endl;
	}
}

// 粒子初始化
void particle_init(particle* particles, int _num_particle, MatND hist)
{
	for (int i = 0; i < _num_particle; i++)
	{
		//所有粒子初始化到框中的目标中心
		particles[i].x = roiRect.x + 0.5 * roiRect.width;
		particles[i].y = roiRect.y + 0.5 * roiRect.height;
		particles[i].xPre = particles[i].x;
		particles[i].yPre = particles[i].y;
		particles[i].xOri = particles[i].x;
		particles[i].yOri = particles[i].y;
		//pParticles->rect = roiRect;
		particles[i].width = roiRect.width;
		particles[i].height = roiRect.height;
		particles[i].scale = 1.0;
		particles[i].scalePre = 1.0;
		particles[i].hist = hist;
		//权重全部为0？
		particles[i].weight = 0;
	}
}

// 粒子状态转移（新位置生成预测）
particle transition(particle p, int w, int h, gsl_rng* rng)
{
	//double rng_nu_x = rng.uniform(0., 1.);
	//double rng_nu_y = rng.uniform(0., 0.5);
	float x, y, s;
	particle pn;

	/* sample new state using second-order autoregressive dynamics */
	x = A1 * (p.x - p.xOri) + A2 * (p.xPre - p.xOri) +
		B0 * gsl_ran_gaussian(rng, TRANS_X_STD)/*rng.gaussian(TRANS_X_STD)*/ + p.xOri;  //计算该粒子下一时刻的x
	pn.x = MAX(0.0, MIN((float)w - 1.0, x));
	y = A1 * (p.y - p.yOri) + A2 * (p.yPre - p.yOri) +
		B0 * gsl_ran_gaussian(rng, TRANS_Y_STD)/*rng.gaussian(TRANS_Y_STD)*/ + p.yOri;
	pn.y = MAX(0.0, MIN((float)h - 1.0, y));
	s = A1 * (p.scale - 1.0) + A2 * (p.scalePre - 1.0) +
		B0 * gsl_ran_gaussian(rng, TRANS_S_STD)/*rng.gaussian(TRANS_S_STD)*/ + 1.0;
	pn.scale = MAX(0.1, s);
	pn.xPre = p.x;
	pn.yPre = p.y;
	pn.scalePre = p.scale;
	pn.xOri = p.xOri;
	pn.yOri = p.yOri;
	pn.width = p.width;
	pn.height = p.height;
	//pn.hist = p.hist;
	pn.weight = 0;

	return pn;
}

// 粒子权重归一化
void normalize_weights(particle* particles, int n)
{
	float sum = 0;
	int i;

	for (i = 0; i < n; i++)
		sum += particles[i].weight;
	for (i = 0; i < n; i++)
		particles[i].weight /= sum;
}

// 粒子权重排序
int particle_cmp(const void* p1, const void* p2)
{
	//这个函数配合qsort，如果这个函数返回值: (1) <0时：p1排在p2前面   (2)  >0时：p1排在p2后面
	particle* _p1 = (particle*)p1;
	particle* _p2 = (particle*)p2;
	//这里就由大到小排序了
	return _p2->weight - _p1->weight;
}

// 粒子重采样
void resample(particle* particles, particle* new_particles, int num_particles)
{
	//计算每个粒子的概率累计和
	double sum[NUM_PARTICLE], temp_sum = 0;
	int k = 0;
	for (int j = num_particles - 1; j >= 0; j--) {
		temp_sum += particles[j].weight;
		sum[j] = temp_sum;
	}
	//为每个粒子生成一个均匀分布【0，1】的随机数
	RNG sum_rng(time(NULL));
	double Ran[NUM_PARTICLE];
	for (int j = 0; j < num_particles; j++) {
		sum_rng = sum_rng.next();
		Ran[j] = sum_rng.uniform(0., 1.);
	}
	//在粒子概率累积和数组中找到最小的大于给定随机数的索引，复制该索引的粒子一次到新的粒子数组中 【从权重高的粒子开始】
	for (int j = 0; j < num_particles; j++) {
		int copy_index = get_min_index(sum, num_particles, Ran[j]);
		new_particles[k++] = particles[copy_index];
		if (k == num_particles)
			break;
	}
	//如果上面的操作完成，新粒子数组的数量仍少于原给定粒子数量，则复制权重最高的粒子，直到粒子数相等
	while (k < num_particles)
	{
		new_particles[k++] = particles[0]; //复制权值最高的粒子
	}
	//以新粒子数组覆盖久的粒子数组
	for (int i = 0; i < num_particles; i++)
	{
		particles[i] = new_particles[i];  //复制新粒子到particles
	}
}

// 二分法求数组中大于给定值的最小值索引
int get_min_index(double *array, int length, double _value)
{
	int _index = (length - 1) / 2;
	int last_index = length - 1;
	int _index_up_limit = length - 1;
	int _index_down_limit = 0;
	// 先判断极值
	if (array[0] <= _value) {
		return 0;
	}
	if (array[length - 1] > _value) {
		return length - 1;
	}
	for (; _index != last_index;) {
		last_index = _index;
		if (array[_index] > _value) {
			_index = (_index_up_limit + _index) / 2;
			_index_down_limit = last_index;
		}
		else if (array[_index] < _value) {
			_index = (_index_down_limit + _index) / 2;
			_index_up_limit = last_index;
		}
		else if (array[_index] == _value) {
			_index--;
			break;
		}
	}
	return _index;
}
