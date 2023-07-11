#include <iostream>
#include <opencv2/opencv.hpp>
#include "segmentation/segmentation.hpp"


using namespace std;
using namespace cv;

Rect2d selection;
bool isDragging = false;
Mat image;


//vFunction(n_links and some t_links use) coefficient, also the max_value of vFunction, set bigger then Boundary weighs more
const double super_parameter_gamma = 50.0;
//decide the maxIterTime of loop
const int super_parameter_maxIterTimes = 40;
//test image name
const string test_image_name = "people1";

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDOWN) {
        // 开始拖拽
        isDragging = true;
        selection.x = x;
        selection.y = y;
    }
    else if (event == EVENT_LBUTTONUP) {
        // 停止拖拽
        isDragging = false;
        selection.width = x - selection.x;
        selection.height = y - selection.y;
        // 存储矩形
        *(Rect2d*)userdata = selection;
        // 绘制矩形
        rectangle(image, selection, Scalar(0, 255, 0), 2);
        imshow("Image", image);
    }
    else if (event == EVENT_MOUSEMOVE && isDragging) {
        // 更新矩形的大小
        selection.width = x - selection.x;
        selection.height = y - selection.y;
        // 绘制矩形
        Mat tempImage = image.clone();
        rectangle(tempImage, selection, Scalar(0, 255, 0), 2);
        imshow("Image", tempImage);
    }
}

void grabcut_test(Mat &img, Rect2d &interst_area){
    Segmentation segmentation(img,super_parameter_gamma,super_parameter_maxIterTimes);
    segmentation.initByRect(interst_area);
    segmentation.iter();
    Mat fgd;
    segmentation.getFgdImg(fgd);
    imshow("my_grabcut_result",fgd);
    waitKey(0);
}

void opencv_grabcut(Mat &img, Rect2d &roi){
    Mat bgdModel, fgdModel;
    Mat mask, bg_mask, fg_mask;
    Mat result;
    double t1 = cv::getTickCount();
    grabCut(img,mask,roi,bgdModel,fgdModel,1,GC_INIT_WITH_RECT);
    double t2 = cv::getTickCount();
    std::cout <<"opencv_consume_time: "<< (t2 - t1) / cv::getTickFrequency() * 1000 << "ms" << endl;


    bg_mask = (mask==1)+(mask==3);
    fg_mask = (mask==0)+(mask==2);
    img.copyTo(result,bg_mask);
//    imshow("opencv_grabcut_mask",bg_mask);
    imshow("opencv_grabcut_result",result);
//    waitKey(0);
    return;
}

int main(){
    image = imread("/home/nuc/workspace/GrabCut/data/"+test_image_name+".jpg");
//    resize(image,image,Size(400,300));
    Mat img = image.clone();
    if (image.empty()) {
        std::cout << "无法读取图像文件" << std::endl;
        return -1;
    }

    // 创建窗口
    namedWindow("Image");

    // 设置鼠标事件回调函数的参数
    Rect2d selectedRect(0,0,0,0);
    setMouseCallback("Image", mouseCallback, &selectedRect);

    // 显示图像并等待选择矩形
    imshow("Image", image);
    waitKey(0);

    while (selectedRect.x == 0);
    // 打印选择的矩形坐标

    std::cout << "Selected Rect: (" << selectedRect.x << ", " << selectedRect.y << ", "
              << selectedRect.width << ", " << selectedRect.height << ")" << std::endl;
    opencv_grabcut(img,selectedRect);
    grabcut_test(img,selectedRect);
    return 0;
}