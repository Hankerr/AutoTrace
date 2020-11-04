#include "AutoTrace.h"
// AutoTrace.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。


int main()
{
	// 环境配置测试
	//*opencv库环境配置*/
	// 读入一张图片（poyanghu缩小图）    
	//Mat img = imread(".\\starry-sky.jpg");
	// 创建一个名为 "图片"窗口    
	//namedWindow("图片");
	// 在窗口中显示图片   
	//imshow("图片", img);
	// 等待6000 ms后窗口自动关闭    
	//waitKey(6000);

	/*gsl库环境配置*/
	//double z = gsl_sf_bessel_J0(0.5);
	//cout << z << endl;

	// 参数初始化
	// 粒子数变量初始化
	int num_particles = NUM_PARTICLE;
	PARTICLE particles[NUM_PARTICLE];
	PARTICLE new_particles[NUM_PARTICLE];
	PARTICLE * pParticles;
	pParticles = particles;
	// 随机指针
	gsl_rng* rng;
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rng, time(NULL));

	// 定义相关变量
	float s;
	int j;
	// 初始化对象
	AutoTrace autoTrace;
	autoTrace.video_name = "material.avi";
	// 视频元定义
	Mat frame, hsv_frame;
	vector<Mat> frames;
	// 目标的直方图
	MatND hist;
	// 视频素材
	VideoCapture capture(autoTrace.video_name.c_str());
	// 判断视频是否打开
	if (!capture.isOpened())
	{
		cout << "Video Open Fail!" << endl;
		system("pause");
		return 0;
	}

	// 读取视频
	while (1)
	{
		//cout << "Reading Video..." << endl;
		capture >> frame;
		if (frame.empty()) {
			cout << "Read Video Finish" << endl;
			break;
		}
		// 创建新窗口
		cv::namedWindow("frame", WINDOW_NORMAL);
		// 复制一个原始帧，给框定目标回调函数用
		current_frame = frame.clone();
		cv::setMouseCallback("frame", MouseEvent);
		frames.push_back(frame.clone());
		cv::imshow("frame", frame);
		cv::waitKey(40);

		// 鼠标选中相应区域进行区域粒子初始化
		if (getTargetFlag == true) {
			// 目标区域转换到hsv空间
			cv::cvtColor(roiImage, hsv_roiImage, COLOR_BGR2HSV);
			// 计算目标区域的直方图
			cv::calcHist(&hsv_roiImage, 1, channels, Mat(), hist, 2, histSize, ranges);
			cv::normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());	// 归一化L2
			//粒子初始化
			autoTrace.particle_init(particles, num_particles, hist);
		}
		else {
			// 未选取区域继续执行
			cout << "Not region Extraction，Pls doing..." << endl;
			continue;
		}
		// 初始化完成，进行粒子处理
		while (1)
		{
			capture >> frame;
			cout << "Init Frame Finish" << endl;
			break;
		}
		current_frame = frame.clone();
		frames.push_back(current_frame);
		// 粒子的操作
		for (j = 0; j < num_particles; j++)
		{
			// 这里利用高斯分布的随机数来生成每个粒子下一次的位置以及范围
			particles[j] = autoTrace.transition(particles[j], frame.cols, frame.rows, rng);
			double s = particles[j].scale;
			// 根据新生成的粒子信息截取对应frame上的区域
			int new_x = std::max(
				0,
				std::min(
					cvRound(particles[j].x - 0.5*particles[j].width),
					cvRound(frame.cols - particles[j].width*s)
				)
			);
			int new_y = std::max(
				0,
				std::min(
					cvRound(particles[j].y - 0.5*particles[j].height),
					cvRound(frame.rows - particles[j].height*s)
				)
			);
			cout << "Fresh_x1_y1:" << new_x << "_" << new_y << ",new_x2_y2:" << cvRound(particles[j].width*s) << "_" << cvRound(particles[j].height*s) << endl;
			Rect imgParticleRect = Rect(new_x, new_y, cvRound(particles[j].width*s), cvRound(particles[j].height*s));
			Mat imgParticle = current_frame(imgParticleRect).clone();
			// 上述区域转换到hsv空间
			cv::cvtColor(imgParticle, imgParticle, COLOR_BGR2GRAY);// COLOR_BGR2GRAY->CV_BGR2HSV
			// 计算区域的直方图
			// 异常点
			cout << "channels:" << channels << " particles[j].hist:" << particles[j].hist << " phistSize:" << histSize << " ranges:" << ranges << endl;
			system("pause");
			cv::calcHist(&imgParticle, 1, channels, Mat(), particles[j].hist, 2, histSize, ranges);
			// 直方图归一化到（0，1）
			cv::normalize(particles[j].hist, particles[j].hist, 0, 1, NORM_MINMAX, -1, Mat());	// 归一化L2
			// 画出蓝色的粒子框
			cv::rectangle(frame, imgParticleRect, Scalar(255, 0, 0), 1, 8);
			//cv::imshow("particle", imgParticle);
			// 比较目标的直方图和上述计算的区域直方图,更新particle权重
			particles[j].weight = exp(-100 * (compareHist(hist, particles[j].hist, CV_COMP_BHATTACHARYYA)));
			int jj = 0;
		}
		// 粒子归一化权重 
		autoTrace.normalize_weights(particles, num_particles);
		int k = 0;
		// 粒子按权重从高到低排序
		std::qsort(particles, num_particles, sizeof(particle), AutoTrace::particle_cmp);
		// 重采样
		autoTrace.resample(particles, new_particles, num_particles);

		// 需要修改
		// 这里直接取权重最高的作为目标了，标准做法应该是按加权平均来计算目标位置
		s = particles[0].scale;
		int new_x = std::max(
			0,
			std::min(
				cvRound(particles[0].x - 0.5*particles[0].width),
				cvRound(frame.cols - particles[0].width*s)
			)
		);
		int new_y = std::max(
			0,
			std::min(
				cvRound(particles[0].y - 0.5*particles[0].height),
				cvRound(frame.rows - particles[0].height*s)
			)
		);
		Rect rectTrackingTemp = Rect(new_x, new_y, cvRound(particles[0].width*s), cvRound(particles[0].height*s));
		cv::rectangle(frame, rectTrackingTemp, Scalar(0, 0, 255), 1, 8, 0);
		cv::imshow("frame", frame);
		cv::waitKey(400);
	}
	return 0;
}

// 粒子初始化
void AutoTrace::particle_init(particle* particles, int _num_particle, MatND hist)
{
	for (int i = 0; i < _num_particle; i++) {
		// 所有粒子初始化到框中的目标中心
		particles[i].x = roiRect.x + 0.5 * roiRect.width;
		particles[i].y = roiRect.y + 0.5 * roiRect.height;
		particles[i].xPre = particles[i].x;
		particles[i].yPre = particles[i].y;
		particles[i].xOri = particles[i].x;
		particles[i].yOri = particles[i].y;
		// pParticles->rect = roiRect;
		particles[i].width = roiRect.width;
		particles[i].height = roiRect.height;
		particles[i].scale = 1.0;
		particles[i].scalePre = 1.0;
		particles[i].hist = hist;
		// 权重全部为0
		particles[i].weight = 0;
	}
}

// 粒子状态转移（新位置生成预测）
particle AutoTrace::transition(particle p, int w, int h, gsl_rng * rng)
{
	particle pn;
	/* sample new state using second-order autoregressive dynamics */
	// 计算该粒子下一时刻的x
	double x = A1 * (p.x - p.xOri) + A2 * (p.xPre - p.xOri) + B0 * gsl_ran_gaussian(rng, TRANS_X_STD) + p.xOri;
	pn.x = MAX(0.0, MIN((float)w - 1.0, x));
	double y = A1 * (p.y - p.yOri) + A2 * (p.yPre - p.yOri) + B0 * gsl_ran_gaussian(rng, TRANS_Y_STD) + p.yOri;
	pn.y = MAX(0.0, MIN((float)h - 1.0, y));
	double s = A1 * (p.scale - 1.0) + A2 * (p.scalePre - 1.0) + B0 * gsl_ran_gaussian(rng, TRANS_S_STD) + 1.0;
	pn.scale = MAX(0.1, s);
	pn.xPre = p.x;
	pn.yPre = p.y;
	pn.scalePre = p.scale;
	pn.xOri = p.xOri;
	pn.yOri = p.yOri;
	pn.width = p.width;
	pn.height = p.height;
	pn.weight = 0;
	return pn;
}

// 粒子权重归一化
void AutoTrace::normalize_weights(particle *particles, int n)
{
	double sum = 0;
	int i;
	for (i = 0; i < n; i++)
		sum += particles[i].weight;
	for (i = 0; i < n; i++)
		particles[i].weight /= sum;
}

// 粒子重采样
void AutoTrace::resample(particle *particles, particle *new_particles, int num_particles)
{
	// 计算每个粒子的概率累计和
	double sum[NUM_PARTICLE], temp_sum = 0;
	int k = 0;
	for (int j = num_particles - 1; j >= 0; j--) {
		temp_sum += particles[j].weight;
		sum[j] = temp_sum;
	}
	// 为每个粒子生成一个均匀分布[0，1]的随机数
	RNG sum_rng(time(NULL));
	double Ran[NUM_PARTICLE];
	for (int j = 0; j < num_particles; j++) {
		sum_rng = sum_rng.next();
		Ran[j] = sum_rng.uniform(0., 1.);
	}
	// 在粒子概率累积和数组中找到最小的大于给定随机数的索引，复制该索引的粒子一次到新的粒子数组中 【从权重高的粒子开始】
	for (int j = 0; j < num_particles; j++) {
		int copy_index = AutoTrace::get_min_index(sum, num_particles, Ran[j]);
		new_particles[k++] = particles[copy_index];
		if (k == num_particles)
			break;
	}
	// 如果上面的操作完成，新粒子数组的数量仍少于原给定粒子数量，则复制权重最高的粒子，直到粒子数相等
	while (k < num_particles)
	{
		new_particles[k++] = particles[0]; //复制权值最高的粒子
	}
	// 以新粒子数组覆盖久的粒子数组
	for (int i = 0; i < num_particles; i++)
	{
		// 复制新粒子到particles
		particles[i] = new_particles[i];
	}
}

// 粒子权重排序
int AutoTrace::particle_cmp(const void* p1, const void* p2)
{
	//这个函数配合qsort，如果这个函数返回值: (1) <0时：p1排在p2前面   (2)  >0时：p1排在p2后面
	particle* _p1 = (particle*)p1;
	particle* _p2 = (particle*)p2;
	//这里就由大到小排序了
	return _p2->weight - _p1->weight;
}

// 获取最小下标
int AutoTrace::get_min_index(double * array, int length, double _value)
{
	int _index = (length - 1) / 2;
	int last_index = length - 1;
	int _index_up_limit = length - 1;
	int _index_down_limit = 0;
	//先判断极值
	if (array[0] <= _value) {
		return 0;
	}
	if (array[length - 1] > _value) {
		return length - 1;
	}
	for (; _index != last_index;) {
		//cout << _index << endl;
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
	//cout << "final result:" << endl;
	//cout << _index << endl;
	return _index;
}

// 鼠标事件截取目标
void MouseEvent(int event, int x, int y, int flags, void * win_name)
{
	try
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
			cout << "选取区域x1_y2:" << startPoint.x << "_" << startPoint.y << ",x2_y2:" << endPoint.x << "_" << endPoint.y << endl;
		}
		//显示区域
		if (downFlag == true && upFlag == false) {
			Point tempPoint;
			tempPoint.x = x;
			tempPoint.y = y;
			Mat tempImage = current_frame.clone();//取原图像复制
			//用矩形标记
			rectangle(tempImage, startPoint, tempPoint, Scalar(0, 0, 255), 2, 3, 0);
			// 调试打开
			// imshow((char*)win_name, tempImage);
		}

		//按下选取完并弹起后
		if (downFlag == true && upFlag == true) {
			//起点和终点不相同时，才提取区域
			if (startPoint.x != endPoint.x && startPoint.y != endPoint.y) {
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
		}
	}
	catch (const Exception& e) {
		const char* errorStr = cvErrorStr(e.code);
		cout << errorStr << endl;
		system("pause");
		exit(0);
	}
}

// 提取感兴趣Mat元
Mat regionExtraction(int xRoi, int yRoi, int widthRoi, int heightRoi)
{
	//创建与原图像同大小的Mat
	Mat roiImage;// (srcImage.rows, srcImage.cols, CV_8UC3);
	//提取感兴趣区域
	roiImage = current_frame(Rect(xRoi, yRoi, widthRoi, heightRoi)).clone();
	// 调试打开
	// cv::imshow("set-roi", roiImage);
	return roiImage;
}
