//
// Created by nuc on 23-7-6.
//
#pragma once

#ifndef MAIN_GMM_H
#define MAIN_GMM_H
#define SHOW_KMEANS_CENTER

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

    double calculatePointProbability(const cv::Vec3b& point, int k);

    void calculateParm(int total_points,  vector<vector<Vec3b>> cluster_sets);
public:
    GMM(int K);

    void init_parm_by_KMeans(vector<Vec3b> samples_vec);

    int getRelatedK(Vec3b point);

    double getWeightedProb(Vec3b point);

    void update_parm(vector<Vec3b> samples_vec, vector<int> &data_labels);
};


#endif //MAIN_GMM_H
