#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat img = imread("/home/lz/code/SeamCarving-master/pics/inputA.jpg"), M;
    cvtColor(img, M, COLOR_BGR2GRAY);

    // cout << (int)*min_element(M.ptr<uchar>(M.rows - 1), M.ptr<uchar>(M.rows - 1) + M.cols) << endl;

    Mat mats[10];
    for (int i = 0; i < 10; i++)
        mats[i] = img.rowRange(0, img.rows - 1).clone();
    namedWindow("test");
    imshow("test", img);
    waitKey(-1);
    return 0;
}