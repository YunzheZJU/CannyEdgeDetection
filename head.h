//
// Created by Yunzhe on 2017/12/13.
//

#ifndef OPENCV_A1_HEAD_H
#define OPENCV_A1_HEAD_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat Candy(const Mat &frame);

//******************Sobel算子计算X、Y方向梯度和梯度方向角********************
//第一个参数imageSourc原始灰度图像；
//第二个参数imageSobelX是X方向梯度图像；
//第三个参数imageSobelY是Y方向梯度图像；
//第四个参数pointDrection是梯度方向角数组指针
//*************************************************************
void SobelGradDirction(Mat imageSource, Mat &imageSobelX, Mat &imageSobelY, double *&pointDrection);

//******************计算Sobel的X和Y方向梯度幅值*************************
//第一个参数imageGradX是X方向梯度图像；
//第二个参数imageGradY是Y方向梯度图像；
//第三个参数SobelAmpXY是输出的X、Y方向梯度图像幅值
//*************************************************************
void SobelAmplitude(Mat imageGradX, const Mat &imageGradY, Mat &SobelAmpXY);

//******************局部极大值抑制*************************
//第一个参数imageInput输入的Sobel梯度图像；
//第二个参数imageOutPut是输出的局部极大值抑制图像；
//第三个参数pointDirection是图像上每个点的梯度方向数组指针
//*************************************************************
void LocalMaxValue(Mat imageInput, Mat &imageOutput, double *pointDirection);

//******************双阈值处理*************************
//第一个参数imageInput输入和输出的的Sobel梯度幅值图像；
//第二个参数lowThreshold是低阈值
//第三个参数highThreshold是高阈值
//******************************************************
void DoubleThreshold(Mat &imageInput, Mat &lowOutput, Mat &highOutput, double lowThreshold, double highThreshold);

//******************双阈值中间像素连接处理*********************
//第一个参数imageInput输入和输出的的Sobel梯度幅值图像；
//第二个参数lowThreshold是低阈值
//第三个参数highThreshold是高阈值
//*************************************************************
void DoubleThresholdLink(Mat &imageInput, double lowThreshold, double highThreshold);

#endif //OPENCV_A1_HEAD_H
