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
    get_energy_img(img, energy_img, ENERGY_FUN_SOBEL_L1); // Calculate the energy img of original img.
    find_k_seams(energy_img, seams, atoi(argv[3]) - img.cols, 'v'); // Find vertical seams with lowest energy cost.
    while(!seams.empty()) {
        add_seam<Vec3b>(img, seams.front(), 'v'); // Add vertical seam.
        imshow("Image Enlarging Example", img);
        waitKey(10); // Sleep 0.01s to show the process of seam carving.
        seams.pop();
    }
    get_energy_img(img, energy_img, ENERGY_FUN_SOBEL_L1); // Calculate the energy img of original img.
    find_k_seams(energy_img, seams, atoi(argv[4]) - img.rows, 'h');
    while(!seams.empty()) {
        add_seam<Vec3b>(img, seams.front(), 'h'); // Remove horizontal seam.
        imshow("Image Enlarging Example", img);
        waitKey(10);
        seams.pop();
    }

    imwrite(argv[2], img);
    waitKey(-1);
    return 0;
}