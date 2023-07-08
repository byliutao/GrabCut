//
// Created by nuc on 23-7-6.
//
#pragma once

#ifndef MAIN_GMM_H
#define MAIN_GMM_H

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class GMM {
private:
    int _K;
    vector<double> _weights;
    vector<Vec3f> _means;
    vector<Mat> _covs;

    vector<Vec3f> *_cluster_sets;

    double calculatePointProbability(const cv::Vec3f& point, int k);

    void calculateParm(int total_points);
public:
    GMM(int K);

    void init_parm_by_KMeans(vector<Vec3f> samples_vec);

    int getRelatedK(Vec3f point);

    int getProbTimeWeight(Vec3f point, int k);

    void update_parm(vector<Vec3f> samples_vec, vector<int> &data_labels);
};


#endif //MAIN_GMM_H
