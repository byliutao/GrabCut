#include <iostream>
#include <opencv2/opencv.hpp>
#include "segmentation/segmentation.hpp"
using namespace std;
using namespace cv;

Rect2d selection;
bool isDragging = false;
Mat image;

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

Mat calcGrayHist(const Mat & image)
{
    Mat histogram = Mat::zeros(Size(256,1),CV_32SC1);    	 //256对应的是0~255共计256个像素值
    //注意，Size对应的是x和y，也就是第一个元数是矩阵的列数
    int rows = image.rows;   	 //输入图像的行数
    int cols = image.cols;		 //输入图像的列数

    for(int r =0;r<rows;r++)
    {
        for(int c = 0;c<cols;c++)
        {
            int index = (int)image.at<uchar>(r,c);	//获取每个点的像素值
            histogram.at<int>(0,index) +=1;			//获取了一个像素值，在相应的位置上加1
        }
    }
    return histogram;
}

void grabcut_test(Mat &img, Rect2d &interst_area){
    Segmentation segmentation(img);
    segmentation.initByRect(interst_area);
    segmentation.iter();
    Mat fgd;
    segmentation.getFgdImg(fgd);
    imshow("fgd",fgd);
    waitKey(0);
}

void opencv_grabcut(Mat &img, Rect2d &roi){
    Mat bgdModel, fgdModel;
    Mat mask, bg_mask, fg_mask;
    Mat result;

    grabCut(img,mask,roi,bgdModel,fgdModel,1,GC_INIT_WITH_RECT);

//    Mat mask_hist = calcGrayHist(mask);
//    std::vector<int> array;
//    for(int i = 0; i < mask_hist.cols; i++){
//        array.push_back(mask_hist.at<int>(0,i));
//    }

    bg_mask = (mask==1)+(mask==3);
    fg_mask = (mask==0)+(mask==2);
    img.copyTo(result,bg_mask);
    imshow("opencv_grabcut_mask",bg_mask);
    imshow("opencv_grabcut_result",result);
//    waitKey(0);
    return;
}

int main(){
    image = imread("/home/nuc/workspace/GrabCut/data/fox.jpg");
    resize(image,image,Size(480,320));
    Mat img = image.clone();
    if (image.empty()) {
        std::cout << "无法读取图像文件" << std::endl;
        return -1;
    }

    // 创建窗口
    namedWindow("Image");

    // 设置鼠标事件回调函数的参数
    Rect2d selectedRect;
    setMouseCallback("Image", mouseCallback, &selectedRect);

    // 显示图像并等待选择矩形
    imshow("Image", image);
    waitKey(0);

    // 打印选择的矩形坐标
    std::cout << "Selected Rect: (" << selectedRect.x << ", " << selectedRect.y << ", "
              << selectedRect.width << ", " << selectedRect.height << ")" << std::endl;
    opencv_grabcut(img,selectedRect);
    grabcut_test(img,selectedRect);
    return 0;
}