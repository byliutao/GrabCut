//
// Created by nuc on 23-7-6.
//

#include "GMM.h"
double GMM::calculatePointProbability(const cv::Vec3b& point, int k)
{
    cv::Mat diff = ( cv::Mat_<float>(1,3) << point[0] - _means[k][0], point[1] - _means[k][1], point[2] - _means[k][2]);
    cv::Mat diffT;
    cv::transpose(diff, diffT);
//    double t1 = cv::getTickCount();
    cv::Mat exponent = -0.5 * diff * _covs_inv[k] * diffT;
    double probability = (1.0 / (std::sqrt(std::pow(2 * CV_PI, 3) * _covs_det[k]))) * std::exp(exponent.at<float>(0, 0));
//    double t2 = cv::getTickCount();
//    std::cout <<"time: "<< (t2 - t1) / cv::getTickFrequency() * 1000 << " ";
    return probability;
}

double GMM::calculatePointProbabilitySpeedVersion(const cv::Vec3b& point, int k){
//    double t1 = cv::getTickCount();
    Vec3f diff = (Vec3f)point - _means[k];
    double exponent = diff[0] * (diff[0] * _covs_inv[k].at<float>(0, 0) + diff[1] * _covs_inv[k].at<float>(1, 0) + diff[2] * _covs_inv[k].at<float>(2, 0))
                    + diff[1] * (diff[0] * _covs_inv[k].at<float>(0, 1) + diff[1] * _covs_inv[k].at<float>(1, 1) + diff[2] * _covs_inv[k].at<float>(2, 1))
                    + diff[2] * (diff[0] * _covs_inv[k].at<float>(0, 2) + diff[1] * _covs_inv[k].at<float>(1, 2) + diff[2] * _covs_inv[k].at<float>(2, 2));
    double probability = (1.0 / (std::sqrt(std::pow(2 * CV_PI, 3) * _covs_det[k]))) * exp(-0.5 * exponent);
//    double t2 = cv::getTickCount();
//    std::cout <<"timeS: "<< (t2 - t1) / cv::getTickFrequency() * 1000 << " ";
    return probability;
}


void GMM::calculateParm(int total_points, vector<vector<Vec3b>> cluster_sets) {
    _means.clear();
    _weights.clear();
    _covs.clear();
    _covs_inv.clear();
    _covs_det.clear();
    for (int i = 0; i < _K; i++) {
        Vec3f mean(0.0, 0.0, 0.0);
        cv::Mat covMat(3, 3, CV_32FC1, cv::Scalar(0));
        int cluster_size = cluster_sets[i].size();

        //计算均值
        for (int j = 0; j < cluster_sets[i].size(); j++) {
            mean += (Vec3f)cluster_sets[i][j];
        }
        mean /= (float)cluster_size;

        //计算协方差
        for (int j = 0; j < cluster_sets[i].size(); j++) {
            cv::Vec3f diff = (Vec3f)cluster_sets[i][j] - mean;
            Mat mult = (cv::Mat_<float>(3,3) <<  diff[0]*diff[0], diff[0]*diff[1], diff[0]*diff[2],
                                                            diff[1]*diff[0], diff[1]*diff[1], diff[1]*diff[2],
                                                             diff[2]*diff[0], diff[2]*diff[1], diff[2]*diff[2]);
            covMat += mult;
        }

        covMat /= (cluster_size - 1);


        _means.push_back(mean);
        _weights.push_back((float)cluster_size / (float)total_points);
        _covs.push_back(covMat);
        _covs_inv.push_back(covMat.inv());
        _covs_det.push_back(determinant(covMat));
    }
}

GMM::GMM(int K){
    _K = K;
}

void GMM::init_parm_by_KMeans(vector<Vec3b> samples_vec){
    vector<Vec3f> samples_vec_f;
    for(int i = 0; i < samples_vec.size(); i++){
        samples_vec_f.emplace_back((Vec3f)samples_vec[i]);
    }
    Mat samples((int)samples_vec_f.size(), 3, CV_32FC1, &samples_vec_f[0][0]);


    // 设置K-means聚类的参数
    int numClusters = _K;
    cv::TermCriteria termCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.1);
    int attempts = 3;
    int flags = cv::KMEANS_RANDOM_CENTERS;

    cv::Mat labels, centers;
    cv::kmeans(samples, numClusters, labels, termCriteria, attempts, flags, centers);

    // 保存分类好的点集
    vector<vector<Vec3b>> cluster_sets;
    for(int i = 0; i < _K; i++){
        vector<Vec3b> tmp;
        cluster_sets.push_back(tmp);
    }

    for (int i = 0; i < samples_vec.size(); i++) {
        cluster_sets[labels.at<int>(i,0)].push_back(samples_vec[i]);
    }

    calculateParm(samples_vec.size(),cluster_sets);

#ifdef SHOW_KMEANS_CENTER

#endif
}

int GMM::getRelatedK(Vec3b point){
    int resK = 0;
    double max_prob = 0;
    for(int k = 0; k < _K; k++){
        double tmp_prob = calculatePointProbabilitySpeedVersion(point,k);
        if(tmp_prob > max_prob){
            max_prob = tmp_prob;
            resK = k;
        }
    }
    return resK;
}

double GMM::getWeightedProb(Vec3b point){
    double res = 0;
    for(int i = 0; i < _K; i++){
        res += calculatePointProbabilitySpeedVersion(point,i)*_weights[i];
    }
    return res;
}

void GMM::update_parm(const vector<Vec3b> &samples_vec, const vector<int> &data_labels){
    // 保存分类好的点集
    vector<vector<Vec3b>> cluster_sets;
    for(int i = 0; i < _K; i++){
        vector<Vec3b> tmp;
        cluster_sets.push_back(tmp);
    }
    for (int i = 0; i < samples_vec.size(); i++) {
        cluster_sets[data_labels[i]].push_back(samples_vec[i]);
    }
    calculateParm(samples_vec.size(),cluster_sets);


}