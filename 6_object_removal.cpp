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
    if(argc != 4) {
        cout << "Usage: ./a <input_jpg> <mask_jpg> <optput_jpg>" << endl;
        return 0;
    }

    Mat img = imread(argv[1]), mask = imread(argv[2], IMREAD_GRAYSCALE), energy_img;
    int old_width = img.cols;
    double min_val = 0;
    stack<Point> seam;
    namedWindow("Object Removal");

    // Seam carving according to the mask.
    while (min_val < UCHAR_MAX / 8) {
        get_energy_img(img, energy_img, mask, ENERGY_FUN_SOBEL_L1); // Calculate the energy img of original img.
        find_vertical_seam(energy_img, &seam); // Find vertical seam with lowest energy cost.
        remove_seam<Vec3b>(img, seam, 'v'); // Remove vertical seam.
        remove_seam<uchar>(mask, seam, 'v');
        minMaxLoc(mask, &min_val, NULL);
        imshow("Object Removal", img);
        waitKey(10); // Sleep 0.01s to show the process of seam carving.
    }
    queue<stack<Point>> seams;
    for(int k = old_width - img.cols; k; ) {
        get_energy_img(img, energy_img, ENERGY_FUN_SOBEL_L1); // Calculate the energy img of original img.
        find_k_seams(energy_img, seams, k, 'v'); // Find vertical seams with lowest energy cost.
        k -= seams.size();
        add_seams<Vec3b>(img, seams, 'v'); // Add vertical seam.
        imshow("Object Removal", img);
        waitKey(10);
    }
    imwrite(argv[3], img);
    waitKey(-1);
    return 0;
}