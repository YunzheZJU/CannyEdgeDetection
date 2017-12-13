#include "head.h"
#include <ctime>

int main() {
    Mat src = imread("lena.png");
    Mat result = Candy(src);
    imshow("My Candy Detection", result);
    waitKey();
    return 0;
}

Mat Candy(const Mat &frame) {
    int time_0;
    int time_1;
//    Normalizing
    Mat imageGray;
    cvtColor(frame, imageGray, CV_BGR2GRAY);
//    Filtering
    time_0 = clock();
    Mat imageGaussion;
    GaussianBlur(imageGray, imageGaussion, Size(3, 3), 0, 0);
    time_1 = clock();
    cout << "1. Filtering takes " << time_1 - time_0 << " milliseconds." << endl;
//    Enhancing
    Mat imageGradientY;
    Mat imageGradientX;
    double *pointDirection; //定义梯度方向角数组
    time_0 = clock();
    GenerateGradient(imageGaussion, imageGradientX, imageGradientY, pointDirection);  //计算X、Y方向梯度和方向角
    time_1 = clock();
    cout << "2. GenerateGradient takes " << time_1 - time_0 << " milliseconds." << endl;

    Mat imageGradient;
    time_0 = clock();
    CombineGradient(imageGradientX, imageGradientY, imageGradient);   //计算X、Y方向梯度融合幅值
    time_1 = clock();
    cout << "3. CombineGradient takes " << time_1 - time_0 << " milliseconds." << endl;
//    Detecting
    Mat imageNMS;
    time_0 = clock();
    NMS(imageGradient, imageNMS, pointDirection);  //局部非极大值抑制
    time_1 = clock();
    cout << "4. NMS takes " << time_1 - time_0 << " milliseconds." << endl;

    Mat imageLowThreshold;
    Mat imageHighThreshold;
    time_0 = clock();
    SplitWithThreshold(imageNMS, imageLowThreshold, imageHighThreshold, 70, 120);        //双阈值处理
    time_1 = clock();
    cout << "5. SplitWithThreshold takes " << time_1 - time_0 << " milliseconds." << endl;

    Mat imageCandy;
    time_0 = clock();
    LinkEdge(imageCandy, imageLowThreshold, imageHighThreshold);
    time_1 = clock();
    cout << "6. DoubleThresholdLink takes " << time_1 - time_0 << " milliseconds." << endl;
//    Done
//    imshow("Original", frame);
//    imshow("Gaussian Blur", imageGaussion);
//    imshow("Gradient X", imageGradientX);
//    imshow("Gradient Y", imageGradientY);
//    imshow("Combine Gradient X and Y", imageGradient);
//    imshow("NMS Image", imageNMS);
//    imshow("Low Threshold", imageLowThreshold);
//    imshow("High Threshold", imageHighThreshold);
//    imshow("Combine Low and High Threshold", imageNMS);
    return imageCandy;
}

void GenerateGradient(const Mat &imageSource, Mat &imageSobelX, Mat &imageSobelY, double *&pointDirection) {
    int time_0;
    int time_1;
    time_0 = clock();
    pointDirection = new double[(imageSource.rows - 1) * (imageSource.cols - 1)];
//    for (int i = 0; i < (imageSource.rows - 1) * (imageSource.cols - 1); i++) {
//        pointDirection[i] = 0;
//    }
    imageSobelX = Mat::zeros(imageSource.size(), CV_32SC1);
    imageSobelY = Mat::zeros(imageSource.size(), CV_32SC1);
    time_1 = clock();
    cout << "2. Initializing takes " << time_1 - time_0 << " milliseconds." << endl;

    int step = imageSource.step;
    int stepXY = imageSobelX.step;
    int rowCount = imageSource.rows;
    int columnCount = imageSource.cols;
    time_0 = clock();
    for (int i = 1; i < (rowCount - 1); i++) {
        const uchar *pixelsPreviousRow = imageSource.ptr<uchar>(i - 1);
        const uchar *pixelsThisRow = imageSource.ptr<uchar>(i);
        const uchar *pixelsNextRow = imageSource.ptr<uchar>(i + 1);
        uchar *pixelsThisRow_x = imageSobelX.ptr<uchar>(i);
        uchar *pixelsThisRow_y = imageSobelY.ptr<uchar>(i);
        for (int j = 1, k = 0; j < (columnCount - 1); j++, k++) {
            //通过指针遍历图像上每一个像素
            double gradY = pixelsPreviousRow[j + 1] + pixelsThisRow[j + 1] * 2 + pixelsNextRow[j + 1] -
                           pixelsPreviousRow[j - 1] - pixelsThisRow[j - 1] * 2 - pixelsNextRow[j - 1];
            double gradX = pixelsNextRow[j - 1] + pixelsNextRow[j] * 2 + pixelsNextRow[j + 1] -
                           pixelsPreviousRow[j - 1] - pixelsPreviousRow[j] * 2 - pixelsPreviousRow[j + 1];
            pixelsThisRow_x[j * (stepXY / step)] = static_cast<uchar>(abs(gradX));
            pixelsThisRow_y[j * (stepXY / step)] = static_cast<uchar>(abs(gradY));
            if (gradX != 0) {
                pointDirection[k] = atan(gradY / gradX) * 57.3 + 90;// (- PI / 2, PI / 2)转换到(0, 180)
            } else {
                pointDirection[k] = 180;
            }
        }
    }
    time_1 = clock();
    cout << "2. Calculating takes " << time_1 - time_0 << " milliseconds." << endl;
    convertScaleAbs(imageSobelX, imageSobelX);
    convertScaleAbs(imageSobelY, imageSobelY);
}

void CombineGradient(const Mat &imageGradX, const Mat &imageGradY, Mat &SobelAmpXY) {
    SobelAmpXY = Mat::zeros(imageGradX.size(), CV_32FC1);
    for (int i = 0; i < SobelAmpXY.rows; i++) {
        const uchar *pixelsThisRow_x = imageGradX.ptr<uchar>(i);
        const uchar *pixelsThisRow_y = imageGradY.ptr<uchar>(i);
        float *pixelsThisRow_xy = SobelAmpXY.ptr<float>(i);
        for (int j = 0; j < SobelAmpXY.cols; j++) {
            const uchar xj = pixelsThisRow_x[j];
            const uchar yj = pixelsThisRow_y[j];
            pixelsThisRow_xy[j] = static_cast<float>(sqrt(xj * xj + yj * yj));
        }
    }
    convertScaleAbs(SobelAmpXY, SobelAmpXY);
}

void NMS(const Mat &imageInput, Mat &imageOutput, double *pointDirection) {
    imageOutput = imageInput.clone();
    int rowCount = imageInput.rows;
    int columnCount = imageInput.cols;
    for (int i = 1; i < rowCount - 1; i++) {
        uchar *pixelsPreviousRow = imageOutput.ptr<uchar>(i - 1);
        uchar *pixelsThisRow = imageOutput.ptr<uchar>(i);
        uchar *pixelsNextRow = imageOutput.ptr<uchar>(i + 1);
        for (int j = 1, k = 0; j < columnCount - 1; j++, k++) {
            double tPD = tan(pointDirection[i * (columnCount - 1) + j]);
            double tPD_180 = tan(180 - pointDirection[i * (columnCount - 1) + j]);

            if (pointDirection[k] <= 45) {
                if (pixelsThisRow[j] <=
                    (pixelsThisRow[j + 1] + (pixelsPreviousRow[j + 1] - pixelsThisRow[j + 1]) * tPD) ||
                    (pixelsThisRow[j] <=
                     (pixelsThisRow[j - 1] + (pixelsNextRow[j - 1] - pixelsThisRow[j - 1]) * tPD))) {
                    pixelsThisRow[j] = 0;
                }
            } else if (pointDirection[k] <= 90) {
                if (pixelsThisRow[j] <=
                    (pixelsPreviousRow[j] + (pixelsPreviousRow[j + 1] - pixelsPreviousRow[j]) / tPD) ||
                    pixelsThisRow[j] <= (pixelsNextRow[j] + (pixelsNextRow[j - 1] - pixelsNextRow[j]) / tPD)) {
                    pixelsThisRow[j] = 0;
                }
            } else if (pointDirection[k] <= 135) {
                if (pixelsThisRow[j] <=
                    (pixelsPreviousRow[j] + (pixelsPreviousRow[j - 1] - pixelsPreviousRow[j]) / tPD_180) ||
                    pixelsThisRow[j] <= (pixelsNextRow[j] + (pixelsNextRow[j + 1] - pixelsNextRow[j]) / tPD_180)) {
                    pixelsThisRow[j] = 0;
                }
            } else if (pointDirection[k] <= 180) {
                if (pixelsThisRow[j] <=
                    (pixelsThisRow[j - 1] + (pixelsPreviousRow[j - 1] - pixelsThisRow[j - 1]) * tPD_180) ||
                    pixelsThisRow[j] <= (pixelsThisRow[j + 1] + (pixelsNextRow[j + 1] - pixelsThisRow[j]) * tPD_180)) {
                    pixelsThisRow[j] = 0;
                }
            } else {
                cout << "Invalid pointDirection: " << pointDirection[k] << endl;
            }
        }
    }
}

void SplitWithThreshold(const Mat &imageInput, Mat &lowOutput, Mat &highOutput, double lowThreshold, double highThreshold) {
    lowOutput = imageInput.clone();
    highOutput = imageInput.clone();
    int rowCount = imageInput.rows;
    int columnCount = imageInput.cols;
    for (int i = 0; i < rowCount; i++) {
        const uchar *pixelsThisRow = imageInput.ptr<uchar>(i);
        uchar *pixelsThisRow_low = lowOutput.ptr<uchar>(i);
        uchar *pixelsThisRow_high = highOutput.ptr<uchar>(i);
        for (int j = 0; j < columnCount; j++) {
            uchar pixel = pixelsThisRow[j];
            if (pixel >= highThreshold) {
                pixelsThisRow_high[j] = 255;
            } else {
                pixelsThisRow_high[j] = 0;
                if (pixel <= lowThreshold) {
                    pixelsThisRow_low[j] = 0;
                } else {
                    pixelsThisRow_low[j] = 255;
                }
            }
        }
    }
}

void LinkEdge(Mat &imageOutput, const Mat &lowThresImage, const Mat &highThresImage) {
    imageOutput = highThresImage.clone();
    int rowCount = imageOutput.rows;
    int columnCount = imageOutput.cols;
    // 为计算方便，牺牲图像四周1像素宽的一圈
    for (int i = 1; i < rowCount - 1; i++) {
        uchar *pixelsPreviousRow = imageOutput.ptr<uchar>(i - 1);
        uchar *pixelsThisRow = imageOutput.ptr<uchar>(i);
        uchar *pixelsNextRow = imageOutput.ptr<uchar>(i + 1);
        for (int j = 1; j < columnCount - 1; j++) {
            if (pixelsThisRow[j] == 255) {
                GoAhead(i, j, pixelsPreviousRow, pixelsThisRow, pixelsNextRow, lowThresImage, imageOutput);
            }
            if (pixelsNextRow[j - 1] == 255) {
                GoAhead(i + 1, j - 1, pixelsThisRow, pixelsNextRow, imageOutput.ptr<uchar>(i + 1), lowThresImage,
                        imageOutput);
            }
            if (pixelsNextRow[j] == 255) {
                GoAhead(i + 1, j, pixelsThisRow, pixelsNextRow, imageOutput.ptr<uchar>(i + 1), lowThresImage,
                        imageOutput);
            }
            if (pixelsNextRow[j + 1] == 255) {
                GoAhead(i + 1, j + 1, pixelsThisRow, pixelsNextRow, imageOutput.ptr<uchar>(i + 1), lowThresImage,
                        imageOutput);
            }
        }
    }
}

void
GoAhead(int i, int j, uchar *pixelsPreviousRow, uchar *pixelsThisRow, uchar *pixelsNextRow, const Mat &lowThresImage,
        Mat &imageOutput) {
    // 判断左下方、右方、下方和右下方是否接续
    if (pixelsThisRow[j + 1] != 255 && pixelsNextRow[j + 1] != 255 && pixelsNextRow[j] != 255 &&
        pixelsNextRow[j - 1] != 255) {
        // 若不接续，从低阈值图中查找8领域是否接续，并对左上方、上方、右上方和左上方递归调用自身
        const uchar *pixelsPreviousRow_low = lowThresImage.ptr<uchar>(i - 1);
        const uchar *pixelsThisRow_low = lowThresImage.ptr<uchar>(i);
        const uchar *pixelsNextRow_low = lowThresImage.ptr<uchar>(i + 1);
        // 左上
        if (pixelsPreviousRow_low[j - 1] == 255) {
            pixelsPreviousRow[j - 1] = 255;
            if (i != 0 && j != 0) {
                GoAhead(i - 1, j - 1, imageOutput.ptr<uchar>(i - 1), pixelsPreviousRow, pixelsThisRow, lowThresImage,
                        imageOutput);
            }
        }
        // 上
        if (pixelsPreviousRow_low[j] == 255) {
            pixelsPreviousRow[j] = 255;
            if (i != 0) {
                GoAhead(i - 1, j, imageOutput.ptr<uchar>(i - 1), pixelsPreviousRow, pixelsThisRow, lowThresImage,
                        imageOutput);
            }
        }
        // 右上
        if (pixelsPreviousRow_low[j + 1] == 255) {
            pixelsPreviousRow[j + 1] = 255;
            if (i != 0 && j != imageOutput.cols) {
                GoAhead(i - 1, j + 1, imageOutput.ptr<uchar>(i - 1), pixelsPreviousRow, pixelsThisRow, lowThresImage,
                        imageOutput);
            }
        }
        // 左
        if (pixelsThisRow_low[j - 1] == 255) {
            pixelsThisRow[j - 1] = 255;
            if (i != 0 && j != 0) {
                GoAhead(i - 1, j - 1, imageOutput.ptr<uchar>(i - 1), pixelsPreviousRow, pixelsThisRow, lowThresImage,
                        imageOutput);
            }
        }
        // 右
        if (pixelsThisRow_low[j + 1] == 255) {
            pixelsThisRow[j + 1] = 255;
        }
        // 左下
        if (pixelsNextRow_low[j - 1] == 255) {
            pixelsNextRow[j - 1] = 255;
        }
        // 下
        if (pixelsNextRow_low[j] == 255) {
            pixelsNextRow[j] = 255;
        }
        // 右下
        if (pixelsNextRow_low[j + 1] == 255) {
            pixelsNextRow[j + 1] = 255;
        }
    }
}