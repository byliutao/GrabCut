//
// Created by nuc on 23-7-6.
//

#include "GMM.h"
double GMM::calculatePointProbability(const cv::Vec3f& point, int k)
{
    cv::Mat diff = cv::Mat(point - _means[k]).reshape(1, 3);
    cv::Mat diffT;
    cv::transpose(diff, diffT);
    cv::Mat exponent = -0.5 * diff * _covs[k].inv() * diffT;
    double det = cv::determinant(_covs[k]);
    double probability = (1.0 / (std::sqrt(std::pow(2 * CV_PI, 3) * det))) * std::exp(exponent.at<float>(0, 0));
    return probability;
}

void GMM::calculateParm(int total_points) {
    for (int i = 0; i < _K; i++) {
        Vec3f mean(0.0, 0.0, 0.0);
        cv::Mat covMat(3, 3, CV_32FC1, cv::Scalar(0));
        int cluster_size = _cluster_sets[i].size();

        //计算均值
        for (int j = 0; j < _cluster_sets[i].size(); j++) {
            mean += _cluster_sets[i][j];
        }
        mean /= cluster_size;

        //计算协方差
        for (int j = 0; j < _cluster_sets[i].size(); j++) {
            cv::Vec3f diff = _cluster_sets[i][j] - mean;
            cv::Mat diffMat = cv::Mat(diff).reshape(1, 3);
            cv::Mat diffMatT;
            cv::transpose(diffMat, diffMatT);
            covMat += diffMat * diffMatT;
        }
        covMat /= (cluster_size - 1);

        _means.push_back(mean);
        _weights.push_back(cluster_size / total_points);
        _covs.push_back(covMat);
    }
}

GMM::GMM(int K){
    _K = K;
    _cluster_sets = new vector<Vec3f>[_K];
}

void GMM::init_parm_by_KMeans(vector<Vec3f> samples_vec){
    Mat samples((int)samples_vec.size(), 3, CV_32FC1, &samples_vec[0][0]);


    // 设置K-means聚类的参数
    int numClusters = _K;
    cv::TermCriteria termCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.1);
    int attempts = 3;
    int flags = cv::KMEANS_RANDOM_CENTERS;

    cv::Mat labels, centers;
    cv::kmeans(samples, numClusters, labels, termCriteria, attempts, flags, centers);

    // 保存分类好的点集
    for (int i = 0; i < samples_vec.size(); i++) {
        _cluster_sets[labels.at<int>(i,0)].push_back(samples_vec[i]);
    }

    calculateParm(samples_vec.size());

}

int GMM::getRelatedK(Vec3f point){
    int resK = 0;
    int max_prob = 0;
    for(int k = 0; k < _K; k++){
        int tmp_prob = calculatePointProbability(point,k);
        if(tmp_prob > max_prob){
            max_prob = tmp_prob;
            resK = k;
        }
    }
    return resK;
}

int GMM::getProbTimeWeight(Vec3f point, int k){
    return calculatePointProbability(point,k)*_weights[k];
}

void GMM::update_parm(vector<Vec3f> samples_vec, vector<int> &data_labels){
    // 保存分类好的点集
    for (int i = 0; i < samples_vec.size(); i++) {
        _cluster_sets[data_labels[i]].push_back(samples_vec[i]);
    }

    calculateParm(samples_vec.size());
}