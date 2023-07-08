//#pragma once

#ifndef MAIN_SEGMENTATION_H
#define MAIN_SEGMENTATION_H

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "GMM.h"
#include "graph.h"

using namespace std;
using namespace cv;

class Segmentation{
private:
    Mat _source_img;
    Mat _T_U;
    const double _gamma = 50.0;
    double _beta;
    const double _lambda = 450.0; //_lambda = 9*_gamma

    void assignGMM(GMM &fgd, GMM &bgd, Mat &img_k);

    void learGMM(GMM &fgd, GMM &bgd, const vector<Vec3f> &fgd_vec, const vector<Vec3f> &bgd_vec, Mat &img_k);

    void estimateSeg(GMM &fgd, GMM &bgd, Mat &img_k);

    void getFgdBgdKbyImgK(vector<int> &fgd_labels, vector<int> &bgd_labels, Mat &img_k);

    void getFgdBgdVecByTU(vector<Vec3f> &fgd_vec, vector<Vec3f> &bgd_vec);

    bool isContainByImage(int i, int j);

    void calculateBeta();

    double getMaxCap(const vector<Vec2i> neighbors_location, const vector<pair<Vec3b,uchar>> &neighbors, const Vec2i center_location, const pair<Vec3b,uchar> &center);

public:
    typedef enum
    {
        Object	= 0,
        Background	= 1,
        UnknownObject = 2,
        UnknownBackground = 3
    } pixelType;

    Segmentation(Mat &source_img);

    void initByRect(Rect2d rect);

    void iter();

};

#endif