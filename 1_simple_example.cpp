#include <iostream>
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
    namedWindow("Seam carving simple example");

    stack<Point> seam;
    for (int i = 0; i < img.cols - atoi(argv[3]); i++) {
        get_energy_img(img, energy_img, ENERGY_FUN_SOBEL_L1); // Calculate the energy img of original img.
        find_vertical_seam(energy_img, &seam); // Find vertical seam with lowest energy cost.
        remove_seam<Vec3b>(img, seam, 'v'); // Remove vertical seam.
        imshow("Seam carving simple example", img);
        waitKey(10); // Sleep 0.01s to show the process of seam carving.
    }
    for (int i = 0; i < img.rows - atoi(argv[4]); i++) {
        get_energy_img(img, energy_img, ENERGY_FUN_SOBEL_L1);
        find_horizontal_seam(energy_img, &seam);
        remove_seam<Vec3b>(img, seam, 'h'); // Remove horizontal seam.
        imshow("Seam carving simple example", img);
        waitKey(10);
    }

    imwrite(argv[2], img);
    waitKey(-1);
    return 0;
}