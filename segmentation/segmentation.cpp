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

#ifdef SHOW
    Mat cluster_img(_source_img.size(), _source_img.type());
    Mat mask_img(_source_img.size(), _source_img.type());
    for (int i = 0; i < rows; i++) {
        Vec3b* ptr = _source_img.ptr<Vec3b>(i);
        Vec3b* ptr_c = cluster_img.ptr<Vec3b>(i);
        Vec3b* ptr_m = mask_img.ptr<Vec3b>(i);
        uchar* ptr_u = _T_U.ptr<uchar>(i);
        int* ptr_k = img_k.ptr<int>(i);
        for (int j = 0; j < cols; j++) {
            Vec3b& pixel = ptr[j];
            uchar& value_u = ptr_u[j];
            int& k = ptr_k[j];

            if(value_u == Object || value_u == UnknownObject){
                ptr_c[j] = Vec3b(k*50 % 255,k*70 % 255,0);
                ptr_m[j] = Vec3b(value_u*20 % 255,value_u*30 % 255,  255);
            }
            else{
                ptr_c[j] = Vec3b(k*50 % 255,k*70 % 255,255);
                ptr_m[j] = Vec3b(value_u*50 % 255,value_u*70 % 255,  0);
            }

        }
    }
    imshow("k_cluster_distribution",cluster_img);
    imshow("fgd_bgd_distribution",mask_img);
    waitKey(0);
#endif
}

void Segmentation::learGMM(GMM &fgd, GMM &bgd, Mat &img_k){
    vector<Vec3b> fgd_vec, bgd_vec;
    vector<int> fgd_labels, bgd_labels;
    getFgdBgdInfo(fgd_vec,bgd_vec,fgd_labels,bgd_labels,img_k);
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

    int add_edge_num = 0;
    //generate graph
    for (int i = 0; i < _source_img.rows; ++i) {
        for (int j = 0; j < _source_img.cols; ++j) {
            Vec3b pixel = _source_img.at<cv::Vec3b>(i, j);
            uchar value_u = _T_U.at<uchar>(i,j);
            int k = img_k.at<int>(i,j);

            int node_index = graph.add_node();

            //Set n-links
            if(isContainByImage(i-1,j-1)){

                pixelType leftPt, upPt, leftUpPt, centerPt;
                Vec3b leftPixel, upPixel, leftUpPixel, centerPixel;
                leftUpPt = (pixelType)_T_U.at<uchar>(i-1,j-1);
                leftPt = (pixelType)_T_U.at<uchar>(i,j-1);
                upPt = (pixelType)_T_U.at<uchar>(i-1,j);
                centerPt = (pixelType)_T_U.at<uchar>(i,j);
                leftUpPixel = _source_img.at<cv::Vec3b>(i-1, j-1);
                leftPixel = _source_img.at<cv::Vec3b>(i, j-1);
                upPixel = _source_img.at<cv::Vec3b>(i-1, j);
                centerPixel = _source_img.at<cv::Vec3b>(i, j);

                //(i,j) to (i-1,j-1)    V(or B in iccv01) function, according to the paper equation(11)
                double cap1 = vFunction(centerPt,leftUpPt,centerPixel,leftUpPixel);
                graph.add_edge(node_index,node_index-_source_img.cols-1,cap1,cap1);
                //(i-1,j) to (i-1,j-1)
                double cap2 = vFunction(upPt,leftUpPt,upPixel,leftUpPixel);
                graph.add_edge(node_index-_source_img.cols,node_index-_source_img.cols-1,cap2,cap2);
                //(i,j-1) to (i-1,j-1)
                double cap3 = vFunction(leftPt,leftUpPt,leftPixel,leftUpPixel);
                graph.add_edge(node_index-1,node_index-_source_img.cols-1,cap3,cap3);
                //(i,j-1) to (i-1,j)
                double cap4 = vFunction(leftPt,upPt,leftPixel,upPixel);
                graph.add_edge(node_index-1,node_index-_source_img.cols,cap4,cap4);

                add_edge_num += 4;
                if(j+1 == _source_img.cols){
                    //i,j) to ((i-1,j)
                    double cap5 = vFunction(upPt,centerPt,upPixel,centerPixel);
                    graph.add_edge(node_index,node_index-_source_img.cols,cap5,cap5);
                    add_edge_num ++;
                }
                if(i+1 == _source_img.rows){
                    //(i,j) to (i,j-1)
                    double cap6 = vFunction(leftPt,centerPt,leftPixel,centerPixel);
                    graph.add_edge(node_index,node_index-1,cap6,cap6);
                    add_edge_num ++;
                }
            }

            //Set t-links
            double cap_source = 0.0, cap_sink = 0.0;

            if(value_u == Background){
                cap_sink = _K;
                cap_source = 0;
            }
            else if(value_u == Object){
                cap_sink = 0;
                cap_source = _K;
            }
            else{
                //U(or R in iccv01) function, according to the paper equation(8)
                cap_sink = -log(fgd.getWeightedProb(pixel)) * _lambda;
                cap_source = -log(bgd.getWeightedProb(pixel)) * _lambda;
            }
            graph.add_tweights(node_index, cap_source, cap_sink);
            add_edge_num += 2;
        }
    }

    graph.maxflow();

//    cout<<"preset_edges_num: "<<edges_num<<" real_add_edge: "<<add_edge_num<<endl;
    CV_Assert(edges_num == add_edge_num);
    int rows = _T_U.rows;
    int cols = _T_U.cols;

// 指针遍历图像
    for (int i = 0; i < rows; i++) {
        uchar* ptr = _T_U.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            uchar& value_u = ptr[j];
//            if(1){
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

void Segmentation::getFgdBgdInfo(vector<Vec3b> &fgd_vec, vector<Vec3b> &bgd_vec, vector<int> &fgd_labels, vector<int> &bgd_labels, Mat &img_k){
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
                bgd_vec.push_back(pixel);
                bgd_labels.push_back(k);
            }
            else{
                fgd_vec.push_back(pixel);
                fgd_labels.push_back(k);
            }
        }
    }
}

bool Segmentation::isContainByImage(int i, int j) const{
    int imageWidth = _source_img.cols;
    int imageHeight = _source_img.rows;

    return i >= 0 && i < imageHeight && j >= 0 && j < imageWidth;

}

//according to the equation (5)
void Segmentation::calculateBeta(){
    double totalDiff = 0;
    int total_num = (_source_img.rows - 1) * _source_img.cols
                    + (_source_img.cols - 1) * _source_img.rows
                    + (_source_img.rows - 1) * (_source_img.cols - 1) * 2;
    int add_num = 0;
    for (int i = 0; i < _source_img.rows; ++i) {
        for (int j = 0; j < _source_img.cols; ++j) {
            // 访问像素值
            cv::Vec3b pixel = _source_img.at<cv::Vec3b>(i, j);
            if(isContainByImage(i,j+1)){
                totalDiff += calculateSquareDis(pixel,_source_img.at<cv::Vec3b>(i,j+1));
            }
            if(isContainByImage(i+1,j)){
                totalDiff += calculateSquareDis(pixel,_source_img.at<cv::Vec3b>(i+1,j));
            }
            if(isContainByImage(i+1,j+1)){
                totalDiff += calculateSquareDis(pixel,_source_img.at<cv::Vec3b>(i+1,j+1));
            }
            if(isContainByImage(i+1,j) && isContainByImage(i,j+1)){
                totalDiff += calculateSquareDis(_source_img.at<cv::Vec3b>(i+1,j),_source_img.at<cv::Vec3b>(i,j+1));
            }
        }
    }
    double expectation_diff = totalDiff / (double)total_num;
    _beta = 1.0 / ((expectation_diff) * 2);
}

//according to essay iccv01.pdf
void Segmentation::calculateK(){
    double max_K = 0;
    for (int i = 0; i < _source_img.rows; ++i) {
        for (int j = 0; j < _source_img.cols; ++j) {
            Vec3b pixel = _source_img.at<cv::Vec3b>(i, j);
            uchar value_u = _T_U.at<uchar>(i,j);
            vector<pair<Vec3b,uchar>> neighbors;
            pair<Vec3b,uchar> center(pixel,value_u);

            for(int n = i - 1; n <= i + 1; n++){
                for(int m = j - 1; m <= j + 1; m++){
                    if(isContainByImage(n,m) && (n != i || m != j)){
                        neighbors.emplace_back(_source_img.at<cv::Vec3b>(n, m),_T_U.at<uchar>(n,m));
                    }
                }
            }
            double currentTotalCap = getTotalCap(neighbors, center);
            if(currentTotalCap > max_K){
                max_K = currentTotalCap;
            }
        }
    }
    _K = max_K + 1;
}

double Segmentation::vFunction(pixelType pixelType1, pixelType pixelType2, Vec3b pixelValue1, Vec3b pixelValue2){
    return _gamma * (isSameLevel(pixelType1, pixelType2) ? 1.0 : 0.0) *
           exp(-1.0 * _beta * calculateSquareDis(pixelValue1, pixelValue2));
}

double Segmentation::vFunction1(pixelType pixelType1, pixelType pixelType2, Vec3b pixelValue1, Vec3b pixelValue2, Vec2i position1, Vec2i position2){
    return _gamma * (isSameLevel(pixelType1, pixelType2) ? 1.0 : 0.0) *
            ( 1.0 / cv::norm(position1,position2)) *
            exp(-1.0 * _beta * calculateSquareDis(pixelValue1, pixelValue2));
}


double Segmentation::getTotalCap(const vector<pair<Vec3b,uchar>> &neighbors, const pair<Vec3b,uchar> &center){
    double totalCap = 0;
    for(int i = 0; i < neighbors.size(); i++){
        double cap = 0;
        pair<Vec3b,uchar> neighbor = neighbors[i];

        cap = vFunction((pixelType)neighbor.second,(pixelType)center.second,neighbor.first,center.first);
        totalCap += cap;
    }
    return totalCap;
}

double Segmentation::calculateSquareDis(Vec3b p1, Vec3b p2){
    double res = (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) + (p1[2]-p2[2])*(p1[2]-p2[2]);
    return res;
}


bool Segmentation::isSameLevel(pixelType pixelType1, pixelType pixelType2){
    if((pixelType1 == Background | pixelType1 == UnknownBackground) &&
       (pixelType2 == Background | pixelType2 == UnknownBackground)){
        return true;
    }
    if((pixelType1 == Object | pixelType1 == UnknownObject) &&
       (pixelType2 == Object | pixelType2 == UnknownObject)){
        return true;
    }
    return false;
}


Segmentation::Segmentation(Mat &source_img, double gamma, double lambda, int iter_times){
    _source_img = source_img.clone();
    _gamma = gamma;
    _lambda = lambda;
    _iter_times = iter_times;
}

void Segmentation::initByRect(cv::Rect2d rect) {
    _T_U = Mat::zeros(_source_img.size(), CV_8UC1);
    int rows = _T_U.rows;
    int cols = _T_U.cols;

// 指针遍历图像
    for (int i = 0; i < rows; i++) {
        uchar* ptr = _T_U.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            uchar& value_u = ptr[j];
            if(j>=rect.x && j<=(rect.x+rect.width) && i>=rect.y && i<=(rect.y+rect.height)){
                value_u = UnknownObject;
            }
            else{
                value_u = Background;
            }
        }
    }
}

void Segmentation::iter(){
    GMM fgd(5), bgd(5);
    //获取前后景数据
    vector<Vec3b> fgd_vec, bgd_vec;
    Mat img_k(_source_img.size(), CV_32SC1);

    calculateBeta();
//    calculateK();

    getFgdBgdVecByTU(fgd_vec,bgd_vec);
    fgd.init_parm_by_KMeans(fgd_vec);
    bgd.init_parm_by_KMeans(bgd_vec);
    calculateK();


    for(int i = 0; i < _iter_times; i++){
        double t1 = cv::getTickCount();
        assignGMM(fgd,bgd,img_k);
        double t2 = cv::getTickCount();
        learGMM(fgd,bgd,img_k);
        double t3 = cv::getTickCount();
        estimateSeg(fgd,bgd,img_k);
        double t4 = cv::getTickCount();
        cout<<"iter"<<i<<"  ";
        std::cout <<"setp1: "<< (t2 - t1) / cv::getTickFrequency() * 1000 << " ";
        std::cout <<"setp2: "<< (t3 - t2) / cv::getTickFrequency() * 1000 << " ";
        std::cout <<"setp3: "<< (t4 - t3) / cv::getTickFrequency() * 1000 << " ";
        std::cout <<"total_consume_time: "<< (t4 - t1) / cv::getTickFrequency() * 1000 << "ms ";
        std::cout << std::endl;
    }
}

void Segmentation::getFgdImg(Mat &img){
    Mat bgd_mask = (_T_U==1)+(_T_U==3);
    Mat fgd_mask = (_T_U==0)+(_T_U==2);

    _source_img.copyTo(img,fgd_mask);
    return;
}