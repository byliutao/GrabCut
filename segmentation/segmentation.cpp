#include "segmentation.hpp"

void Segmentation::assignGMM(GMM &fgd, GMM &bgd, Mat &img_k){
    int rows = _source_img.rows;
    int cols = _source_img.cols;
    for (int i = 0; i < rows; i++) {
        Vec3b* ptr = _source_img.ptr<Vec3b>(i);
        uchar* ptr_u = _T_U.ptr<uchar>(i);
        int* ptr_k = img_k.ptr<int>(i);
        for (int j = 0; j < cols; j++) {
            Vec3b& pixel = ptr[j];
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

void Segmentation::learGMM(GMM &fgd, GMM &bgd, vector<Vec3b> &fgd_vec, vector<Vec3b> &bgd_vec, Mat &img_k){
    vector<int> fgd_labels, bgd_labels;
    getFgdBgdKbyImgK(fgd_labels,bgd_labels,img_k);
    fgd.update_parm(fgd_vec,fgd_labels);
    bgd.update_parm(bgd_vec,bgd_labels);
    fgd_vec.clear();
    bgd_vec.clear();
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

            int node_index = graph.add_node();

            pair<Vec3b,uchar> center(pixel,value_u);
            Vec2i center_location(i,j);

            if(isContainByImage(i-1,j-1)){
                //(i,j) to (i-1,j-1)
                double cap1 = _gamma * cv::norm(Vec2i(i-1,j-1),Vec2i(i,j)) *
                             exp(-1.0 * _beta * cv::norm(_source_img.at<cv::Vec3b>(i-1, j-1),_source_img.at<cv::Vec3b>(i, j))) *
                             (_T_U.at<uchar>(i-1,j-1) != _T_U.at<uchar>(i,j)) ? 1.0 : 0.0;
                graph.add_edge(node_index,node_index-_source_img.cols-1,cap1,cap1);
                //(i-1,j) to (i-1,j-1)
                double cap2 = _gamma * cv::norm(Vec2i(i-1,j-1),Vec2i(i-1,j)) *
                              exp(-1.0 * _beta * cv::norm(_source_img.at<cv::Vec3b>(i-1, j-1),_source_img.at<cv::Vec3b>(i-1, j))) *
                              (_T_U.at<uchar>(i-1,j-1) != _T_U.at<uchar>(i-1,j)) ? 1.0 : 0.0;
                graph.add_edge(node_index-_source_img.cols,node_index-_source_img.cols-1,cap2,cap2);
                //(i,j-1) to (i-1,j-1)
                double cap3 = _gamma * cv::norm(Vec2i(i-1,j-1),Vec2i(i,j-1)) *
                              exp(-1.0 * _beta * cv::norm(_source_img.at<cv::Vec3b>(i-1, j-1),_source_img.at<cv::Vec3b>(i, j-1))) *
                              (_T_U.at<uchar>(i-1,j-1) != _T_U.at<uchar>(i,j-1)) ? 1.0 : 0.0;
                graph.add_edge(node_index-1,node_index-_source_img.cols-1,cap3,cap3);
                //(i,j-1) to (i-1,j)
                double cap4 = _gamma * cv::norm(Vec2i(i-1,j),Vec2i(i,j-1)) *
                              exp(-1.0 * _beta * cv::norm(_source_img.at<cv::Vec3b>(i-1, j),_source_img.at<cv::Vec3b>(i, j-1))) *
                              (_T_U.at<uchar>(i-1,j) != _T_U.at<uchar>(i,j-1)) ? 1.0 : 0.0;
                graph.add_edge(node_index-1,node_index-_source_img.cols,cap4,cap4);
                if(j+1 == _source_img.cols){
                    //(i-1,j) to (i,j)
                    double cap1 = _gamma * cv::norm(Vec2i(i-1,j),Vec2i(i,j)) *
                                  exp(-1.0 * _beta * cv::norm(_source_img.at<cv::Vec3b>(i-1, j),_source_img.at<cv::Vec3b>(i, j))) *
                                  (_T_U.at<uchar>(i-1,j) != _T_U.at<uchar>(i,j)) ? 1.0 : 0.0;
                    graph.add_edge(node_index,node_index-_source_img.cols,cap1,cap1);
                }
                if(i+1 == _source_img.rows){
                    //(i,j-1) to (i,j)
                    double cap1 = _gamma * cv::norm(Vec2i(i,j-1),Vec2i(i,j)) *
                                  exp(-1.0 * _beta * cv::norm(_source_img.at<cv::Vec3b>(i, j-1),_source_img.at<cv::Vec3b>(i, j))) *
                                  (_T_U.at<uchar>(i,j-1) != _T_U.at<uchar>(i,j)) ? 1.0 : 0.0;
                    graph.add_edge(node_index,node_index-1,cap1,cap1);
                }
            }

            double W_source, W_sink;
            vector<pair<Vec3b,uchar>> neighbors;
            vector<Vec2i> neighbors_location;

            for(int n = i - 1; n <= i + 1; n++){
                for(int m = j - 1; m <= j + 1; m++){
                    if(isContainByImage(n,m) && (n != i || m != j)){
                        neighbors.emplace_back(_source_img.at<cv::Vec3b>(n, m),_T_U.at<uchar>(n,m));
                        neighbors_location.emplace_back(n,m);
                    }
                }
            }
            double maxCap = getMaxCap(neighbors_location, neighbors, center_location, center);

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
            uchar& value_u = ptr[j];
            if(value_u == UnknownBackground || value_u == UnknownObject){
                if(graph.what_segment(i*_source_img.cols+j) == Graph<double, double, double>::SOURCE){
                    value_u = UnknownObject;
                }
                else{
                    value_u = UnknownBackground;
                }
            }

        }
    }

}

void Segmentation::getFgdBgdKbyImgK(vector<int> &fgd_labels, vector<int> &bgd_labels, Mat &img_k){
    int rows = _source_img.rows;
    int cols = _source_img.cols;
    for (int i = 0; i < rows; i++) {
        Vec3b* ptr = _source_img.ptr<Vec3b>(i);
        int* ptr_k = img_k.ptr<int>(i);
        uchar* ptr_u = _T_U.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            int& k = ptr_k[j];
            uchar& value_u = ptr_u[j];
            Vec3b& pixel = ptr[j];
            if(value_u == UnknownBackground || value_u == Background){
                bgd_labels.push_back(k);
            }
            else{
                fgd_labels.push_back(k);
            }
        }
    }
}

void Segmentation::getFgdBgdVecByTU(vector<Vec3b> &fgd_vec, vector<Vec3b> &bgd_vec){
    int rows = _source_img.rows;
    int cols = _source_img.cols;
    for (int i = 0; i < rows; i++) {
        Vec3b* ptr = _source_img.ptr<Vec3b>(i);
        uchar* ptr_u = _T_U.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            Vec3b& pixel = ptr[j];
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
    int imageWidth = _source_img.cols;
    int imageHeight = _source_img.rows;

    return i >= 0 && i < imageWidth && j >= 0 && j < imageHeight;

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
    vector<Vec3b> fgd_vec, bgd_vec;
    Mat img_k(_source_img.size(), CV_32SC1);

    getFgdBgdVecByTU(fgd_vec,bgd_vec);
    fgd.init_parm_by_KMeans(fgd_vec);
    bgd.init_parm_by_KMeans(bgd_vec);

    for(int i = 0; i < 3; i++){
        assignGMM(fgd,bgd,img_k);
        learGMM(fgd,bgd,fgd_vec,bgd_vec,img_k);
        estimateSeg(fgd,bgd,img_k);
    }
}

void Segmentation::getFgdImg(Mat &img){
    Mat bgd_mask = (_T_U==1)+(_T_U==3);
    Mat fgd_mask = (_T_U==0)+(_T_U==2);

    _source_img.copyTo(img,fgd_mask);
    return;
}