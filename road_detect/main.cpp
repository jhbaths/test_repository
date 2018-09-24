#include <stdio.h>

#include "opencv/cv.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"

using namespace std;
using namespace cv;

class zone_fifo
{
public:
	int head;
	int tail;
	CvPoint* fifo_memory;

	zone_fifo(int depth)
	{
		fifo_memory = (CvPoint*)new int[depth * 2];
		head = 0;
		tail = 0;
	}

	~zone_fifo()
	{
		if (fifo_memory != NULL)
		{
			delete fifo_memory;
		}
	}

	void reset_fifo()
	{
		head = 0;
		tail = 0;
	}

	void fifo_push(CvPoint& pnt)
	{
		if (fifo_memory != NULL)
		{
			fifo_memory[tail] = pnt;
			tail++;
		}
	}

	CvPoint fifo_pop()
	{
		CvPoint rst;

		if ((fifo_memory != NULL) && (head < tail))
		{
			rst = fifo_memory[head];
			head++;
		}
		else
		{
			rst.x = -1;
			rst.y = -1;
		}

		return rst;
	}
};

void get_zone(Mat& input, zone_fifo& zone_detect_fifo, CvPoint& zone_pnt, int class_id)
{
	CvPoint temp;

	zone_detect_fifo.reset_fifo();

	input.at<uchar>(zone_pnt.x, zone_pnt.y) = class_id;
	zone_detect_fifo.fifo_push(zone_pnt);

	while (zone_detect_fifo.head < zone_detect_fifo.tail)
	{
		//从fifo中取出一个点
		temp = zone_detect_fifo.fifo_pop();

		//查看该点周围的四个点，如果是空白点，将其标记为class_id,并加入fifo
		if (input.at<uchar>(temp.x - 1, temp.y) == 0)
		{
			input.at<uchar>(temp.x - 1, temp.y) = class_id;
			zone_detect_fifo.fifo_push(CvPoint(temp.x - 1, temp.y));
		}
		if (input.at<uchar>(temp.x + 1, temp.y) == 0)
		{
			input.at<uchar>(temp.x + 1, temp.y) = class_id;
			zone_detect_fifo.fifo_push(CvPoint(temp.x + 1, temp.y));
		}
		if (input.at<uchar>(temp.x, temp.y - 1) == 0)
		{
			input.at<uchar>(temp.x, temp.y - 1) = class_id;
			zone_detect_fifo.fifo_push(CvPoint(temp.x, temp.y - 1));
		}
		if (input.at<uchar>(temp.x, temp.y + 1) == 0)
		{
			input.at<uchar>(temp.x, temp.y + 1) = class_id;
			zone_detect_fifo.fifo_push(CvPoint(temp.x, temp.y + 1));
		}
	}
}

//在这个图中，255表示分界线，0表示可连通区域
int image_split(Mat& input)
{
	int class_id = 1;
	zone_fifo zone_detect_fifo(input.rows*input.cols);

	//为提高速度，将图像周边一圈像素设置为边界，置为255
	for (int i = 0; i < input.rows; i++)
	{
		input.at<uchar>(i, 0) = 255;
		input.at<uchar>(i, input.cols - 1) = 255;
	}

	for (int i = 0; i < input.cols; i++)
	{
		input.at<uchar>(0, i) = 255;
		input.at<uchar>(input.rows - 1, i) = 255;
	}

	//遍历所有未分类的点
	for (int i = 1; i < input.rows - 1; i++)
	{
		for (int j = 1; j < input.cols - 1; j++)
		{
			if (input.at<uchar>(i, j) == 0)
			{
				//从该点开始进行膨胀，占满该点所在的连通区域
				CvPoint zone_pnt(i, j);
				get_zone(input, zone_detect_fifo, zone_pnt, class_id);

				class_id++;
			}
		}
	}

	return class_id - 1;
}

void set_mask(Mat& img, int ori_value, int new_value)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == ori_value)
			{
				img.at<uchar>(i, j) = new_value;
			}
			else
			{
				img.at<uchar>(i, j) = 0;
			}
		}
	}
}

void get_free_zone(Mat& cur, Mat& last)
{
	int zone_vote[256] = {0};
	int max = 0;
	int zone_id = 0;

	for (int i = 0; i < cur.rows; i++)
	{
		for (int j = 0; j < cur.cols; j++)
		{
			if (last.at<uchar>(i, j) == 1)
			{
				zone_vote[cur.at<uchar>(i, j)]++;
			}

			//int temp = cur.at<uchar>(i, j)* last.at<uchar>(i, j);
			//zone_vote[temp]++;
		}
	}

	for (int i = 1; i < 255; i++)
	{
		if (zone_vote[i] > max)
		{
			max = zone_vote[i];
			zone_id = i;
		}
	}

	set_mask(cur, zone_id, 128);

}

int main()
{
	int64 start, end;
	double freq;
	printf("start to proc!\n");
	freq = getTickFrequency();

	Mat image,result;
	image = imread("1.jpg", 0);

	threshold(image, result, 20, 255, THRESH_BINARY_INV);

	int class_num = image_split(result);

	set_mask(result,3,1);

	Mat last_result = result.clone();

	image = imread("2.jpg", 0);

	start = getTickCount();

	threshold(image, result, 20, 255, THRESH_BINARY_INV);
	class_num = image_split(result); 
	get_free_zone(result, last_result);

	end = getTickCount();

	printf("cycle is:%d!\n", end - start);
	printf("zone time cost is:%f!\n", (end - start) / freq);

	imshow("result", result);

	cvWaitKey(0);

	return 0;
}