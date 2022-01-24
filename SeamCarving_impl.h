#include "SeamCarving.h"

/**
 * @brief Remove the seam from img.
 * Notice that the seam will be empty after operation.
 * @param img Image to be changed.
 * @param seam The seam to remove.
 * @param direction The direction of seam.
 */
template <typename T> void remove_seam(cv::Mat &img, std::stack<cv::Point> &seam, char direction = 'v') {
    while(!seam.empty()) {
        if(direction == 'v')
            for (int j = seam.top().x; j < img.cols - 1; j++)
                img.at<T>(seam.top().y, j) = img.at<T>(seam.top().y, j + 1);
        else if(direction == 'h')
            for (int i = seam.top().y; i < img.rows - 1; i++)
                img.at<T>(i, seam.top().x) = img.at<T>(i + 1, seam.top().x);
        else
            throw std::invalid_argument("unknown direction type.");
        seam.pop();
    }
    img = direction == 'v' ? img.colRange(0, img.cols - 1) : img.rowRange(0, img.rows - 1);
}