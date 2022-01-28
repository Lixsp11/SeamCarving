#include <iostream>
#include <stack>
#include <queue>
#include "SeamCarving.cpp"
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    if(argc != 5) {
        cout << "Usage: ./a <input_jpg> <optput_jpg> <wigth> <height>" << endl;
        return 0;
    }

    Mat img = imread(argv[1]), energy_img;
    namedWindow("Image Enlarging Example");

    queue<stack<Point>> seams;
    for(int k = atoi(argv[3]) - img.cols; k;) {
        get_energy_img(img, energy_img, ENERGY_FUN_SOBEL_L1);           // Calculate the energy img of original img.
        find_k_seams(energy_img, seams, k, 'v'); // Find vertical seams with lowest energy cost.
        k -= seams.size();
        add_seams<Vec3b>(img, seams, 'v'); // Add vertical seam.
        imshow("Image Enlarging Example", img);
        waitKey(10); // Sleep 0.01s to show the process of seam carving.
    }
    for (int k = atoi(argv[4]) - img.rows; k; ) {
        get_energy_img(img, energy_img, ENERGY_FUN_SOBEL_L1); // Calculate the energy img of original img.
        find_k_seams(energy_img, seams, k, 'h');
        k -= seams.size();
        add_seams<Vec3b>(img, seams, 'h'); // Remove horizontal seam.
        imshow("Image Enlarging Example", img);
        waitKey(10);
    }
    imwrite(argv[2], img);
    waitKey(-1);
    return 0;
}