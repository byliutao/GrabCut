#include <iostream>
#include <opencv2/opencv.hpp>
#include <graph.h>
using namespace std;
using namespace cv;

void test(Mat &img, Rect &interst_area){

}


int main(){
    Mat test_img = imread("./data/fox.jpg");
    imshow("test_img",test_img);
    cout<<"hello"<<endl;
    return 0;
}