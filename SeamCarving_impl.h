#include "SeamCarving.h"
#include <opencv4/opencv2/highgui.hpp>

/**
 * @brief Remove the seam from img.
 * @param img Image to be changed.
 * @param seam The seam to remove.
 * @param direction The direction of seam. Use 'v' means vertical or 'h' means horizontal.
 */
template <typename _Tp> void remove_seam(cv::Mat &img, std::stack<cv::Point> seam, char direction = 'v') {
    while(!seam.empty()) {
        if(direction == 'v')
            for (int j = seam.top().x; j < img.cols - 1; j++)
                img.at<_Tp>(seam.top().y, j) = img.at<_Tp>(seam.top().y, j + 1);
        else if(direction == 'h')
            for (int i = seam.top().y; i < img.rows - 1; i++)
                img.at<_Tp>(i, seam.top().x) = img.at<_Tp>(i + 1, seam.top().x);
        else
            throw std::invalid_argument("unknown direction type.");
        seam.pop();
    }
    img = direction == 'v' ? img.colRange(0, img.cols - 1) : img.rowRange(0, img.rows - 1);
}

/**
 * @brief Add the seam to img by averaging them with their left and right neighbors (top and bottom in the horizontal case).
 * @param img Image to be changed.
 * @param seam The seam to remove.
 * @param direction The direction of seam. Use 'v' means vertical or 'h' means horizontal.
 */
template <typename _Tp> void add_seam(cv::Mat &img, std::stack<cv::Point> seam, char direction = 'v') {
    if(direction == 'v') {
        cv::Mat _img = img.clone();
        img.create(_img.rows, _img.cols + 1, _img.type());
        for (int i = 0; i < _img.rows; i++)
            for (int j = 0; j < _img.cols; j++)
                img.at<_Tp>(i, j) = _img.at<_Tp>(i, j);
    }
    else if(direction == 'h') 
        img.resize(img.rows + 1);
    else
        throw std::invalid_argument("unknown direction type.");

    while(!seam.empty()) {
        if(direction == 'v') {
            for (int j = img.cols - 1; j > seam.top().x; j--)
                img.at<_Tp>(seam.top().y, j) = img.at<_Tp>(seam.top().y, j - 1);
            if(seam.top().x > 0 && seam.top().x < img.cols - 2) {
                _Tp scalar1(1), scalar2(1);
                cv::multiply(img.at<_Tp>(seam.top().y, seam.top().x - 1), cv::Scalar(1, 1, 1, 1), scalar1, 0.5);
                cv::multiply(img.at<_Tp>(seam.top().y, seam.top().x + 1), cv::Scalar(1, 1, 1, 1), scalar2, 0.5);
                cv::add(scalar1, scalar2, img.at<_Tp>(seam.top()));
            }
        }
        else {
            for (int i = img.rows - 1; i > seam.top().y; i--)
                img.at<_Tp>(i, seam.top().x) = img.at<_Tp>(i - 1, seam.top().x);
            if(seam.top().y > 0 && seam.top().y < img.rows - 2) {
                _Tp scalar1(1), scalar2(1);
                cv::multiply(img.at<_Tp>(seam.top().y - 1, seam.top().x), cv::Scalar(1, 1, 1, 1), scalar1, 0.5);
                cv::multiply(img.at<_Tp>(seam.top().y + 1, seam.top().x), cv::Scalar(1, 1, 1, 1), scalar2, 0.5);
                cv::add(scalar1, scalar2, img.at<_Tp>(seam.top()));
            }
        }
        seam.pop();
    }
}

template <typename _Tp> void add_seams(cv::Mat &img, std::queue<std::stack<cv::Point>> seams, char direction = 'v') {
    if(direction == 'v') {
        cv::Mat _img = img.clone();
        img.create(_img.rows, _img.cols + seams.size(), _img.type());
        
        // Count all insertion points by row.
        std::vector<std::vector<int>> vv(_img.rows, std::vector<int>());
        while(!seams.empty()) {
            std::stack<cv::Point> seam = seams.front();
            while(!seam.empty()) {
                vv[seam.top().y].push_back(seam.top().x);
                seam.pop();
            }
            seams.pop();
        }

        for (int i = 0; i < _img.rows; i++) {
            std::sort(vv[i].begin(), vv[i].end());
            for (int k = int(vv[i].size()) - 1; k >= -1; k--) {
                int left = (k == -1 ? 0 : vv[i][k]);
                int right = (k == int(vv[i].size()) - 1 ? _img.cols : vv[i][k + 1]);
                for (int j = left; j < right; j++) 
                    img.at<_Tp>(i, j + k + 1) = _img.at<_Tp>(i, j);
                if(k != -1) {
                    if(vv[i][k] == 0)
                        img.at<_Tp>(i, vv[i][k] + k) = _img.at<_Tp>(i, vv[i][k] + 1);
                    else if(vv[i][k] == _img.cols - 1)
                        img.at<_Tp>(i, vv[i][k] + k) = _img.at<_Tp>(i, vv[i][k] - 1);
                    else {
                        _Tp scalar1(1), scalar2(1);
                        cv::multiply(_img.at<_Tp>(i, vv[i][k] - 1), cv::Scalar(1, 1, 1, 1), scalar1, 0.5);
                        cv::multiply(_img.at<_Tp>(i, vv[i][k] + 1), cv::Scalar(1, 1, 1, 1), scalar2, 0.5);
                        cv::add(scalar1, scalar2, img.at<_Tp>(i, vv[i][k] + k));
                    }
                }
            }
        }
    }
    else if (direction == 'h') {
        cv::Mat _img = img.clone();
        img.create(_img.rows + seams.size(), _img.cols, _img.type());
        
        // Count all insertion points by col.
        std::vector<std::vector<int>> vv(_img.cols, std::vector<int>());
        while(!seams.empty()) {
            std::stack<cv::Point> seam = seams.front();
            while(!seam.empty()) {
                vv[seam.top().x].push_back(seam.top().y);
                seam.pop();
            }
            seams.pop();
        }

        for (int j = 0; j < _img.cols; j++) {
            std::sort(vv[j].begin(), vv[j].end());
            for (int k = int(vv[j].size()) - 1; k >= -1; k--) {
                int left = (k == -1 ? 0 : vv[j][k]);
                int right = (k == int(vv[j].size()) - 1 ? _img.rows : vv[j][k + 1]);
                for (int i = left; i < right; i++) 
                    img.at<_Tp>(i + k + 1, j) = _img.at<_Tp>(i, j);
                if(k != -1) {
                    if(vv[j][k] == 0)
                        img.at<_Tp>(vv[j][k] + k, j) = _img.at<_Tp>(vv[j][k] + 1, j);
                    else if(vv[j][k] == _img.rows - 1)
                        img.at<_Tp>(vv[j][k] + k, j) = _img.at<_Tp>(vv[j][k] - 1, j);
                    else {
                        _Tp scalar1(1), scalar2(1);
                        cv::multiply(_img.at<_Tp>(vv[j][k] - 1, j), cv::Scalar(1, 1, 1, 1), scalar1, 0.5);
                        cv::multiply(_img.at<_Tp>(vv[j][k] + 1, j), cv::Scalar(1, 1, 1, 1), scalar2, 0.5);
                        cv::add(scalar1, scalar2, img.at<_Tp>(vv[j][k] + k, j));
                    }
                }
            }
        }
    }
    else
        throw std::invalid_argument("unknown direction type.");
}