#include "segmentation.hpp"

void Segmentation::assignGMM(GMM &fgd, GMM &bgd, Mat &img_k){
    int rows = _source_img.rows;
    int cols = _source_img.cols;
    for (int i = 0; i < rows; i++) {
        Vec3f* ptr = _source_img.ptr<Vec3f>(i);
        uchar* ptr_u = _T_U.ptr<uchar>(i);
        int* ptr_k = img_k.ptr<int>(i);
        for (int j = 0; j < cols; j++) {
            Vec3f& pixel = ptr[j];
            uchar& value_u = ptr_u[j];
            int k = 0;
            if(value_u == UnknownBackground || value_u == Background){
                k = bgd.getRelatedK(pixel);
            }
            else{
                k = fgd.getRelatedK(pixel);
            }
            ptr_k[j] = k;
        }
    }
}

void Segmentation::learGMM(GMM &fgd, GMM &bgd, const vector<Vec3f> &fgd_vec, const vector<Vec3f> &bgd_vec, Mat &img_k){
    vector<int> fgd_labels, bgd_labels;
    getFgdBgdKbyImgK(fgd_labels,bgd_labels,img_k);
    fgd.update_parm(fgd_vec,fgd_labels);
    bgd.update_parm(bgd_vec,bgd_labels);
}

void Segmentation::estimateSeg(GMM &fgd, GMM &bgd, Mat &img_k){
    // according to the paper, V = P | {S, T}
    int nodes_num = _source_img.rows * _source_img.cols;
    // E = N | { {p,S} , {p,T} }
    int edges_num = _source_img.rows * _source_img.cols * 2
                    + (_source_img.rows - 1) * _source_img.cols
                    + (_source_img.cols - 1) * _source_img.rows
                    + (_source_img.rows - 1) * (_source_img.cols - 1) * 2;
    Graph<double,double,double> graph(nodes_num,edges_num);

    //generate graph
    for (int i = 0; i < _source_img.rows; ++i) {
        for (int j = 0; j < _source_img.cols; ++j) {
            Vec3b pixel = _source_img.at<cv::Vec3b>(i, j);
            uchar value_u = _T_U.at<uchar>(i,j);
            int k = img_k.at<int>(i,j);

            vector<pair<Vec3b,uchar>> neighbors;
            vector<Vec2i> neighbors_location;
            pair<Vec3b,uchar> center(pixel,value_u);
            Vec2i center_location(i,j);
            for(int n = i - 1; n <= i + 1; n++){
                for(int m = j - 1; m <= j + 1; m++){
                    if(isContainByImage(n,m) && (n != i || m != j)){
                        neighbors.emplace_back(_source_img.at<cv::Vec3b>(n, m),_T_U.at<uchar>(n,m));
                        neighbors_location.emplace_back(n,m);
                    }
                }
            }
            double maxCap = getMaxCap(neighbors_location, neighbors, center_location, center);

            if(isContainByImage(i,j+1)){
                double cap = _gamma * cv::norm(Vec2i(i,j+1),center_location) *
                             exp(-1.0 * _beta * cv::norm(_source_img.at<cv::Vec3b>(i, j+1),center.first)) *
                             (_T_U.at<uchar>(i,j+1) != center.second) ? 1.0 : 0.0;
                graph.add_edge(i,j+1,cap,cap);
            }
            if(isContainByImage(i+1,j)){
                double cap = _gamma * cv::norm(Vec2i(i+1,j),center_location) *
                             exp(-1.0 * _beta * cv::norm(_source_img.at<cv::Vec3b>(i+1, j),center.first)) *
                             (_T_U.at<uchar>(i+1,j) != center.second) ? 1.0 : 0.0;
                graph.add_edge(i+1,j,cap,cap);
            }
            if(isContainByImage(i+1,j+1)){
                double cap = _gamma * cv::norm(Vec2i(i+1,j+1),center_location) *
                             exp(-1.0 * _beta * cv::norm(_source_img.at<cv::Vec3b>(i+1, j+1),center.first)) *
                             (_T_U.at<uchar>(i+1,j+1) != center.second) ? 1.0 : 0.0;
                graph.add_edge(i+1,j+1,cap,cap);
            }
            if(isContainByImage(i+1,j) && isContainByImage(i,j+1)){
                double cap = _gamma * cv::norm(Vec2i(i+1,j),center_location) *
                             exp(-1.0 * _beta * cv::norm(_source_img.at<cv::Vec3b>(i+1, j),center.first)) *
                             (_T_U.at<uchar>(i+1,j) != center.second) ? 1.0 : 0.0;
                graph.add_edge(i+1,j,cap,cap);
            }

            int node_index = graph.add_node();
            double W_source, W_sink;

            if(value_u == Background){
                W_sink = 1 + maxCap;
                W_source = 0;
            }
            else if(value_u == Object){
                W_sink = 0;
                W_source = 1 + maxCap;
            }
            else{
                W_sink = -log(bgd.getProbTimeWeight(pixel,k));
                W_source = -log(fgd.getProbTimeWeight(pixel,k));
            }
            graph.add_tweights(node_index,W_source,W_sink);
        }
    }

    graph.maxflow();

    int rows = _T_U.rows;
    int cols = _T_U.cols;

// 指针遍历图像
    for (int i = 0; i < rows; i++) {
        uchar* ptr = _T_U.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            uchar& pixel = ptr[j];
            if(pixel == UnknownBackground || pixel == UnknownObject){
                if(graph.what_segment(i*_source_img.cols+j) == Graph<double, double, double>::SOURCE){
                    pixel = UnknownObject;
                }
                else{
                    pixel = UnknownBackground;
                }
            }

        }
    }

}

void Segmentation::getFgdBgdKbyImgK(vector<int> &fgd_labels, vector<int> &bgd_labels, Mat &img_k){
    int rows = _source_img.rows;
    int cols = _source_img.cols;
    for (int i = 0; i < rows; i++) {
        Vec3f* ptr = _source_img.ptr<Vec3f>(i);
        int* ptr_k = img_k.ptr<int>(i);
        uchar* ptr_u = _T_U.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            int& k = ptr_k[j];
            uchar& value_u = ptr_u[j];
            Vec3f& pixel = ptr[j];
            if(value_u == UnknownBackground || value_u == Background){
                bgd_labels.push_back(k);
            }
            else{
                fgd_labels.push_back(k);
            }
        }
    }
}

void Segmentation::getFgdBgdVecByTU(vector<Vec3f> &fgd_vec, vector<Vec3f> &bgd_vec){
    int rows = _source_img.rows;
    int cols = _source_img.cols;
    for (int i = 0; i < rows; i++) {
        Vec3f* ptr = _source_img.ptr<Vec3f>(i);
        uchar* ptr_u = _T_U.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            Vec3f& pixel = ptr[j];
            uchar& value_u = ptr_u[j];
            if(value_u == UnknownBackground || value_u == Background){
                bgd_vec.push_back(pixel);
            }
            else{
                fgd_vec.push_back(pixel);
            }
        }
    }
}

bool Segmentation::isContainByImage(int i, int j){
    return (i < _source_img.rows && j < _source_img.cols);
}

void Segmentation::calculateBeta(){
    double totalDiff = 0;
    int total_num = (_source_img.rows - 1) * _source_img.cols
                    + (_source_img.cols - 1) * _source_img.rows
                    + (_source_img.rows - 1) * (_source_img.cols - 1) * 2;
    for (int i = 0; i < _source_img.rows; ++i) {
        for (int j = 0; j < _source_img.cols; ++j) {
            // 访问像素值
            cv::Vec3b pixel = _source_img.at<cv::Vec3b>(i, j);
            if(isContainByImage(i,j+1)){
                totalDiff += norm(pixel,_source_img.at<cv::Vec3b>(i,j+1));
            }
            if(isContainByImage(i+1,j)){
                totalDiff += norm(pixel,_source_img.at<cv::Vec3b>(i+1,j));
            }
            if(isContainByImage(i+1,j+1)){
                totalDiff += norm(pixel,_source_img.at<cv::Vec3b>(i+1,j+1));
            }
            if(isContainByImage(i+1,j) && isContainByImage(i,j+1)){
                totalDiff += norm(_source_img.at<cv::Vec3b>(i+1,j),_source_img.at<cv::Vec3b>(i,j+1));
            }
        }
    }
    _beta = 1.0 / ((totalDiff / total_num) * 2);
}

double Segmentation::getMaxCap(const vector<Vec2i> neighbors_location, const vector<pair<Vec3b,uchar>> &neighbors, const Vec2i center_location, const pair<Vec3b,uchar> &center){
    double maxCap = 0;
    for(int i = 0; i < neighbors.size(); i++){
        double cap = 0;
        Vec2i location = neighbors_location[i];
        pair<Vec3b,uchar> neighbor = neighbors[i];
        cap = _gamma * cv::norm(location,center_location) *
              exp(-1.0 * _beta * cv::norm(neighbor.first,center.first)) *
              (neighbor.second != center.second) ? 1.0 : 0.0;
        if(cap > maxCap) maxCap = cap;
    }
    return maxCap;
}

Segmentation::Segmentation(Mat &source_img){
    _source_img = source_img.clone();
    calculateBeta();
}

void Segmentation::initByRect(cv::Rect2d rect) {
    _T_U = Mat::zeros(_source_img.size(), CV_8UC1);
    int rows = _T_U.rows;
    int cols = _T_U.cols;

// 指针遍历图像
    for (int i = 0; i < rows; i++) {
        uchar* ptr = _T_U.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            uchar& pixel = ptr[j];
            if(i>=rect.x && i<=(rect.x+rect.height) && j>=rect.y && j<=(rect.y+rect.width)){
                pixel = UnknownObject;
            }
            else{
                pixel = Background;
            }
        }
    }
}

void Segmentation::iter(){
    GMM fgd(5), bgd(5);
    //获取前后景数据
    vector<Vec3f> fgd_vec, bgd_vec;
    Mat img_k(_source_img.size(), CV_32SC1);

    getFgdBgdVecByTU(fgd_vec,bgd_vec);


    for(;;){
        assignGMM(fgd,bgd,img_k);
        learGMM(fgd,bgd,fgd_vec,bgd_vec,img_k);
        estimateSeg(fgd,bgd,img_k);
    }
}