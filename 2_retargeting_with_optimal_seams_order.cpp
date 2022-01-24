#include <iostream>
#include <stack>
#include "SeamCarving.h"
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    if(argc != 5) {
        cout << "Usage: ./a <input_jpg> <optput_jpg> <wigth> <height>" << endl;
        cout << "INFO: Find optimal seams order using dp, and the space complexity of dp matrix is O((m-w)*(n-h)*m*n), ";
        cout << "so carefully set width and height to avoid stack overflow." << endl;
        return 0;
    }

    Mat img = imread(argv[1]), energy_img;
    namedWindow("Retargeting with optimal seams-order");

    stack<char> seams_order; // Data structure stores the seams order, where unstack sequence is the optimal seam sequence.
    get_energy_img(img, energy_img, ENERGY_FUN_SOBEL_L1); // Calculate the energy img of original img.
    find_optimal_seams_order(energy_img, Size(atoi(argv[3]), atoi(argv[4])), seams_order); // Find optimal seams order using dp.
    
    while(!seams_order.empty()) {
        stack<Point> seam;
        get_energy_img(img, energy_img, ENERGY_FUN_SOBEL_L1);
        if(seams_order.top() == 'v') {
            find_vertical_seam(energy_img, &seam);
            remove_seam<Vec3b>(img, seam, 'v');
        }
        else {
            find_horizontal_seam(energy_img, &seam);
            remove_seam<Vec3b>(img, seam, 'h');
        }
        seams_order.pop();
        imshow("Retargeting with optimal seams-order", img);
        waitKey(10); // Sleep 0.01s to show the process of seam carving.
    }
    imwrite(argv[2], img);
    waitKey(-1);
    return 0;
}