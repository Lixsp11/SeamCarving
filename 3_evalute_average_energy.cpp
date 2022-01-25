#include <iostream>
#include <iomanip>
#include "SeamCarving.h"
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    if(argc != 2) {
        cout << "Usage: ./a <input_jpg>" << endl;
        return 0;
    }
    Mat img = imread(argv[1]);
    cout << fixed << setprecision(4) << get_average_energy(img, ENERGY_FUN_SOBEL_L1) << endl; // Keep 4 decimal places
    return 0;
}