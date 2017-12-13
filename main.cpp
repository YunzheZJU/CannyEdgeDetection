#include "head.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    Mat src = imread("1.jpg");
    imshow("Original", src);
    Mat result = Candy(src);
    imshow("My Candy Detection", result);
    waitKey();
    return 0;
}

Mat Candy(Mat &frame) {
    Mat result;
//    Normalizing
    cvtColor(frame, result, CV_BGR2GRAY);
//    Filtering
    GaussianBlur(result, result, Size(3, 3), 0, 0);
//    imshow("After Gaussian Blur", result);
//    Enhancing
    Mat imageSobelY;
    Mat imageSobelX;
    double *pointDirection = new double[(imageSobelX.cols - 1) * (imageSobelX.rows - 1)];  //定义梯度方向角数组
    SobelGradDirction(result, imageSobelX, imageSobelY, pointDirection);  //计算X、Y方向梯度和方向角
//    imshow("Sobel Y",imageSobelY);
//    imshow("Sobel X",imageSobelX);
//    Detecting
    Mat SobelGradAmpl;
    SobelAmplitude(imageSobelX, imageSobelY, SobelGradAmpl);   //计算X、Y方向梯度融合幅值
//    imshow("Soble XYRange", SobelGradAmpl);
    Mat imageLocalMax;
    LocalMaxValue(SobelGradAmpl, imageLocalMax, pointDirection);  //局部非极大值抑制
//    imshow("Non-Maximum Image",imageLocalMax);
    Mat cannyImage;
    cannyImage = Mat::zeros(imageLocalMax.size(), CV_8UC1);
    DoubleThreshold(imageLocalMax, 90, 160);        //双阈值处理
//    imshow("Double Threshold Image",imageLocalMax);
    DoubleThresholdLink(imageLocalMax, 90, 160);   //双阈值中间阈值滤除及连接
//    imshow("Canny Image",imageLocalMax);
//    Done
    return imageLocalMax;
}

void SobelGradDirction(const Mat imageSource, Mat &imageSobelX, Mat &imageSobelY, double *&pointDrection) {
    pointDrection = new double[(imageSource.rows - 1) * (imageSource.cols - 1)];
    for (int i = 0; i < (imageSource.rows - 1) * (imageSource.cols - 1); i++) {
        pointDrection[i] = 0;
    }
    imageSobelX = Mat::zeros(imageSource.size(), CV_32SC1);
    imageSobelY = Mat::zeros(imageSource.size(), CV_32SC1);
    uchar *P = imageSource.data;
    uchar *PX = imageSobelX.data;
    uchar *PY = imageSobelY.data;

    int step = imageSource.step;
    int stepXY = imageSobelX.step;
    int k = 0;
    for (int i = 1; i < (imageSource.rows - 1); i++) {
        for (int j = 1; j < (imageSource.cols - 1); j++) {
            //通过指针遍历图像上每一个像素
            double gradY = P[(i - 1) * step + j + 1] + P[i * step + j + 1] * 2 + P[(i + 1) * step + j + 1] -
                           P[(i - 1) * step + j - 1] - P[i * step + j - 1] * 2 - P[(i + 1) * step + j - 1];
            PY[i * stepXY + j * (stepXY / step)] = static_cast<uchar>(abs(gradY));
            double gradX = P[(i + 1) * step + j - 1] + P[(i + 1) * step + j] * 2 + P[(i + 1) * step + j + 1] -
                           P[(i - 1) * step + j - 1] - P[(i - 1) * step + j] * 2 - P[(i - 1) * step + j + 1];
            PX[i * stepXY + j * (stepXY / step)] = static_cast<uchar>(abs(gradX));
            if (gradX == 0) {
                gradX = 0.00000000000000001;  //防止除数为0异常
            }
            pointDrection[k] = atan(gradY / gradX) * 57.3;//弧度转换为度
            pointDrection[k] += 90;
            k++;
        }
    }
    convertScaleAbs(imageSobelX, imageSobelX);
    convertScaleAbs(imageSobelY, imageSobelY);
}

//******************计算Sobel的X和Y方向梯度幅值*************************
//第一个参数imageGradX是X方向梯度图像；
//第二个参数imageGradY是Y方向梯度图像；
//第三个参数SobelAmpXY是输出的X、Y方向梯度图像幅值
//*************************************************************
void SobelAmplitude(const Mat imageGradX, const Mat &imageGradY, Mat &SobelAmpXY) {
    SobelAmpXY = Mat::zeros(imageGradX.size(), CV_32FC1);
    for (int i = 0; i < SobelAmpXY.rows; i++) {
        for (int j = 0; j < SobelAmpXY.cols; j++) {
            SobelAmpXY.at<float>(i, j) = static_cast<float>(sqrt(
                    imageGradX.at<uchar>(i, j) * imageGradX.at<uchar>(i, j) +
                    imageGradY.at<uchar>(i, j) * imageGradY.at<uchar>(i, j)));
        }
    }
    convertScaleAbs(SobelAmpXY, SobelAmpXY);
}

//******************局部极大值抑制*************************
//第一个参数imageInput输入的Sobel梯度图像；
//第二个参数imageOutPut是输出的局部极大值抑制图像；
//第三个参数pointDirection是图像上每个点的梯度方向数组指针
//*************************************************************
void LocalMaxValue(const Mat imageInput, Mat &imageOutput, double *pointDirection) {
    //imageInput.copyTo(imageOutput);
    imageOutput = imageInput.clone();
    int k = 0;
    for (int i = 1; i < imageInput.rows - 1; i++) {
        for (int j = 1; j < imageInput.cols - 1; j++) {
            int value00 = imageInput.at<uchar>((i - 1), j - 1);
            int value01 = imageInput.at<uchar>((i - 1), j);
            int value02 = imageInput.at<uchar>((i - 1), j + 1);
            int value10 = imageInput.at<uchar>((i), j - 1);
            int value11 = imageInput.at<uchar>((i), j);
            int value12 = imageInput.at<uchar>((i), j + 1);
            int value20 = imageInput.at<uchar>((i + 1), j - 1);
            int value21 = imageInput.at<uchar>((i + 1), j);
            int value22 = imageInput.at<uchar>((i + 1), j + 1);

            if (pointDirection[k] > 0 && pointDirection[k] <= 45) {
                if (value11 <= (value12 + (value02 - value12) * tan(pointDirection[i * (imageOutput.rows - 1) + j])) ||
                    (value11 <=
                     (value10 + (value20 - value10) * tan(pointDirection[i * (imageOutput.rows - 1) + j])))) {
                    imageOutput.at<uchar>(i, j) = 0;
                }
            }
            if (pointDirection[k] > 45 && pointDirection[k] <= 90) {
                if (value11 <= (value01 + (value02 - value01) / tan(pointDirection[i * (imageOutput.cols - 1) + j])) ||
                    value11 <= (value21 + (value20 - value21) / tan(pointDirection[i * (imageOutput.cols - 1) + j]))) {
                    imageOutput.at<uchar>(i, j) = 0;

                }
            }
            if (pointDirection[k] > 90 && pointDirection[k] <= 135) {
                if (value11 <=
                    (value01 + (value00 - value01) / tan(180 - pointDirection[i * (imageOutput.cols - 1) + j])) ||
                    value11 <=
                    (value21 + (value22 - value21) / tan(180 - pointDirection[i * (imageOutput.cols - 1) + j]))) {
                    imageOutput.at<uchar>(i, j) = 0;
                }
            }
            if (pointDirection[k] > 135 && pointDirection[k] <= 180) {
                if (value11 <=
                    (value10 + (value00 - value10) * tan(180 - pointDirection[i * (imageOutput.cols - 1) + j])) ||
                    value11 <=
                    (value12 + (value22 - value11) * tan(180 - pointDirection[i * (imageOutput.cols - 1) + j]))) {
                    imageOutput.at<uchar>(i, j) = 0;
                }
            }
            k++;
        }
    }
}

//******************双阈值处理*************************
//第一个参数imageInput输入和输出的的Sobel梯度幅值图像；
//第二个参数lowThreshold是低阈值
//第三个参数highThreshold是高阈值
//******************************************************
void DoubleThreshold(Mat &imageIput, double lowThreshold, double highThreshold) {
    for (int i = 0; i < imageIput.rows; i++) {
        for (int j = 0; j < imageIput.cols; j++) {
            if (imageIput.at<uchar>(i, j) > highThreshold) {
                imageIput.at<uchar>(i, j) = 255;
            }
            if (imageIput.at<uchar>(i, j) < lowThreshold) {
                imageIput.at<uchar>(i, j) = 0;
            }
        }
    }
}

//******************双阈值中间像素连接处理*********************
//第一个参数imageInput输入和输出的的Sobel梯度幅值图像；
//第二个参数lowThreshold是低阈值
//第三个参数highThreshold是高阈值
//*************************************************************
void DoubleThresholdLink(Mat &imageInput, double lowThreshold, double highThreshold) {
    for (int i = 1; i < imageInput.rows - 1; i++) {
        for (int j = 1; j < imageInput.cols - 1; j++) {
            if (imageInput.at<uchar>(i, j) > lowThreshold && imageInput.at<uchar>(i, j) < 255) {
                if (imageInput.at<uchar>(i - 1, j - 1) == 255 || imageInput.at<uchar>(i - 1, j) == 255 ||
                    imageInput.at<uchar>(i - 1, j + 1) == 255 ||
                    imageInput.at<uchar>(i, j - 1) == 255 || imageInput.at<uchar>(i, j) == 255 ||
                    imageInput.at<uchar>(i, j + 1) == 255 ||
                    imageInput.at<uchar>(i + 1, j - 1) == 255 || imageInput.at<uchar>(i + 1, j) == 255 ||
                    imageInput.at<uchar>(i + 1, j + 1) == 255) {
                    imageInput.at<uchar>(i, j) = 255;
                    DoubleThresholdLink(imageInput, lowThreshold, highThreshold); //递归调用
                } else {
                    imageInput.at<uchar>(i, j) = 0;
                }
            }
        }
    }
}