#include "SeamCarving.h"

/**
 * @brief Remove the seam from img.
 * @note Notice that the seam will be empty after operation.
 * @param img Image to be changed.
 * @param seam The seam to remove.
 * @param direction The direction of seam. Use 'v' means vertical or 'h' means horizontal.
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

/**
 * @brief Add the seam to img by averaging them with their left and right neighbors (top and bottom in the horizontal case).
 * @note Notice that the seam will be empty after operation.
 * @param img Image to be changed.
 * @param seam The seam to remove.
 * @param direction The direction of seam. Use 'v' means vertical or 'h' means horizontal.
 */
template <typename T> void add_seam(cv::Mat &img, std::stack<cv::Point> &seam, char direction = 'v') {
    if(direction == 'v') {
        cv::Mat _img = img.clone();
        img.create(_img.rows, _img.cols + 1, _img.type());
        for (int i = 0; i < _img.rows; i++)
            for (int j = 0; j < _img.cols; j++)
                img.at<T>(i, j) = _img.at<T>(i, j);
    }
    else if(direction == 'h') 
        img.resize(img.rows + 1);
    else
        throw std::invalid_argument("unknown direction type.");

    while(!seam.empty()) {
        if(direction == 'v') {
            for (int j = img.cols - 1; j > seam.top().x; j--)
                img.at<T>(seam.top().y, j) = img.at<T>(seam.top().y, j - 1);
            // if(seam.top().x > 0 && seam.top().x < img.cols - 2) {
            //     cv::add(img.at<T>(seam.top().y, seam.top().x - 1), img.at<T>(seam.top().y, seam.top().x + 1), img.at<T>(seam.top()));
            //     cv::divide(img.at<T>(seam.top()), cv::Scalar(1, 1, 1, 1), img.at<T>(seam.top()));
            // }
        }
        else {
            for (int i = img.rows - 1; i > seam.top().y; i--)
                img.at<T>(i, seam.top().x) = img.at<T>(i - 1, seam.top().x);
            // if(seam.top().y > 0 && seam.top().y < img.rows - 2) {
            //     cv::add(img.at<T>(seam.top().y - 1, seam.top().x), img.at<T>(seam.top().y + 1, seam.top().x), img.at<T>(seam.top()));
            //     cv::divide(2, img.at<T>(seam.top()), img.at<T>(seam.top()));
            // }
        }
        seam.pop();
    }
}