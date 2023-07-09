//
// Created by nuc on 23-7-6.
//

#include "GMM.h"
double GMM::calculatePointProbability(const cv::Vec3b& point, int k)
{
    cv::Mat x = ( cv::Mat_<float>(1,3) << point[0], point[1], point[2] );
    cv::Mat mean = ( cv::Mat_<float>(1, 3) << _means[k][0], _means[k][1], _means[k][2]);
    cv::Mat diff = cv::Mat(x - mean);
    cv::Mat diffT;
    cv::transpose(diff, diffT);
    cv::Mat exponent = -0.5 * diff * _covs[k].inv() * diffT;
    double det = cv::determinant(_covs[k]);
    double probability = (1.0 / (std::sqrt(std::pow(2 * CV_PI, 3) * det))) * std::exp(exponent.at<float>(0, 0));
    return probability;
}

void GMM::calculateParm(int total_points, vector<vector<Vec3b>> cluster_sets) {
    _means.clear();
    _weights.clear();
    _covs.clear();
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
            cv::Mat diffMat = cv::Mat(diff).reshape(1, 3);
            cv::Mat diffMatT;
            cv::transpose(diffMat, diffMatT);
            covMat += diffMat * diffMatT;
        }
        covMat /= (cluster_size - 1);

        _means.push_back(mean);
        _weights.push_back((float)cluster_size / (float)total_points);
        _covs.push_back(covMat);
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
        double tmp_prob = calculatePointProbability(point,k);
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
        res += calculatePointProbability(point,i)*_weights[i];
    }
    return res;
}

void GMM::update_parm(vector<Vec3b> samples_vec, vector<int> &data_labels){
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