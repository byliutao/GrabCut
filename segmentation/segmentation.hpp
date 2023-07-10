//#pragma once

#ifndef MAIN_SEGMENTATION_H
#define MAIN_SEGMENTATION_H
#define SHOW
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "GMM.h"
#include "graph.h"

using namespace std;
using namespace cv;

class Segmentation{
private:
    typedef enum
    {
        Object	= 0,
        Background	= 1,
        UnknownObject = 2,
        UnknownBackground = 3
    } pixelType;

    Mat _source_img;
    Mat _T_U;
    int _iter_times;
    double _gamma;
    double _lambda;
    double _beta;
    double _K;

    void assignGMM(GMM &fgd, GMM &bgd, Mat &img_k);

    void learGMM(GMM &fgd, GMM &bgd, Mat &img_k);

    void estimateSeg(GMM &fgd, GMM &bgd, Mat &img_k);

    void getFgdBgdKbyImgK(vector<int> &fgd_labels, vector<int> &bgd_labels, Mat &img_k);

    void getFgdBgdVecByTU(vector<Vec3b> &fgd_vec, vector<Vec3b> &bgd_vec);

    void getFgdBgdInfo(vector<Vec3b> &fgd_vec, vector<Vec3b> &bgd_vec, vector<int> &fgd_labels, vector<int> &bgd_labels, Mat &img_k);

    bool isContainByImage(int i, int j) const;

    void calculateBeta();

    void calculateK();

    double vFunction(pixelType pixelType1, pixelType pixelType2, Vec3b pixelValue1, Vec3b pixelValue2);

    double vFunction1(pixelType pixelType1, pixelType pixelType2, Vec3b pixelValue1, Vec3b pixelValue2, Vec2i position1, Vec2i position2);

    double getTotalCap(const vector<pair<Vec3b,uchar>> &neighbors, const pair<Vec3b,uchar> &center);

    double calculateSquareDis(Vec3b p1, Vec3b p2);

    bool isSameLevel(pixelType pixelType1, pixelType pixelType2);
public:


    Segmentation(Mat &source_img, double gamma, double lambda, int iter_times);

    void initByRect(Rect2d rect);

    void iter();

    void getFgdImg(Mat &img);

};

#endif