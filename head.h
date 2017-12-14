//
// Created by Yunzhe on 2017/12/13.
//

#ifndef OPENCV_A1_HEAD_H
#define OPENCV_A1_HEAD_H

#include <iostream>
#include <ctime>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define WINDOW_NAME "My Candy Detection"

Mat Candy(const Mat &frame, int lowThreshold, int highThreshold, int kernelSize);

void GenerateGradient(const Mat &imageSource, Mat &imageSobelX, Mat &imageSobelY, double *&pointDirection);

void CombineGradient(const Mat &imageGradX, const Mat &imageGradY, Mat &SobelAmpXY);

void NMS(const Mat &imageInput, Mat &imageOutput, double *pointDirection);

void SplitWithThreshold(const Mat &imageInput, Mat &lowOutput, Mat &highOutput, double lowThreshold, double highThreshold);

void LinkEdge(Mat &imageOutput, const Mat &lowThresImage, const Mat &highThresImage);

void
GoAhead(int i, int j, uchar *pixelsPreviousRow, uchar *pixelsThisRow, uchar *pixelsNextRow, const Mat &lowThresImage,
        Mat &imageOutput);

void onParaChange(int, void *);

void onImageChange(int, void*);

void onSaveImage(int, void*);

#endif //OPENCV_A1_HEAD_H
