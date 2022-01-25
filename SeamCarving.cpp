#include "SeamCarving.h"

void _get_energy_img(cv::Mat const &img, cv::Mat &energy_img, int fun_type  = ENERGY_FUN_SOBEL_L1) {
    if (fun_type == ENERGY_FUN_SOBEL_L1 || fun_type == ENERGY_FUN_SOBEL_L2) {
        cv::Mat gray, dx, dy;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::Sobel(gray, dx, CV_64FC1, 1, 0);
        cv::Sobel(gray, dy, CV_64FC1, 0, 1);
        if (fun_type == ENERGY_FUN_SOBEL_L1) 
            energy_img = cv::abs(dx) + cv::abs(dy);
        else
            cv::magnitude(dx, dy, energy_img);
    }
    else
        throw std::invalid_argument("unknown energy function type.");
}

void get_energy_img(cv::Mat const &img, cv::Mat &energy_img, int fun_type = ENERGY_FUN_SOBEL_L1) {
    _get_energy_img(img, energy_img, fun_type);

    // normalize to [0, 254], use 255 as mask
    double max_val;
    cv::minMaxLoc(energy_img, NULL, &max_val);
    energy_img = energy_img / max_val * (UCHAR_MAX - 1);
    energy_img.convertTo(energy_img, CV_8U);
}

void get_energy_img(cv::Mat const &img, cv::Mat &energy_img, cv::Mat const &mask, int fun_type) {
    _get_energy_img(img, energy_img, fun_type);

    for (int i = 0; i < energy_img.rows; i++)
        for (int j = 0; j < energy_img.cols; j++)
            energy_img.at<double>(i, j) = mask.at<uchar>(i, j) > UCHAR_MAX / 2 ? energy_img.at<double>(i, j) : -UCHAR_MAX;

    // normalize to [0, 254], use 255 as mask
    double min_val, max_val;
    cv::minMaxLoc(energy_img, &min_val, &max_val);
    energy_img = (energy_img - min_val) / (max_val - min_val) * (UCHAR_MAX - 1);
    energy_img.convertTo(energy_img, CV_8U);
}

int find_vertical_seam(cv::Mat const &energy_img, std::stack<cv::Point> *seam = NULL) {
    int M[energy_img.rows][energy_img.cols] = {0};

    // track forward using dp
    for (int j = 0; j < energy_img.cols; j++)
        M[0][j] = (energy_img.at<uchar>(0, j) == UCHAR_MAX ? INT_MAX : energy_img.at<uchar>(0, j));
    for (int i = 1; i < energy_img.rows; i++) {
        for (int j = 0; j < energy_img.cols; j++) {
            int min1 = j == 0 ? INT_MAX : std::min(M[i - 1][j - 1], M[i - 1][j]);
            int min2 = j == energy_img.cols - 1 ? INT_MAX : std::min(M[i - 1][j], M[i - 1][j + 1]);
            // The first case means this pixel is masked, the second means there isn't a path to this pixel.
            if(energy_img.at<uchar>(i, j) == UCHAR_MAX || std::min(min1, min2) == INT_MAX) 
                M[i][j] = INT_MAX;
            else 
                M[i][j] = energy_img.at<uchar>(i, j) + std::min(min1, min2);
        }
    }

    // backtrack to find the seam
    int minLoc = 0;
    int minVal = INT_MAX, ret;
    for (int j = 0; j < energy_img.cols; j++) {
        if(M[energy_img.rows - 1][j] < minVal) {
            minVal = M[energy_img.rows - 1][j];
            minLoc = j;
        }
    }
    ret = minVal;
    // The first case means not cal the seam, the second means can't find a seam.
    if(seam == NULL || minVal == INT_MAX) return 0;
    while (!seam->empty()) seam->pop();
    seam->push(cv::Point(minLoc, energy_img.rows - 1));
    for (int i = energy_img.rows - 1; i > 0; i--) {
        minVal -= energy_img.at<uchar>(i, minLoc);
        if(minLoc > 0 && minVal == M[i - 1][minLoc - 1])
            minLoc--;
        else if(minLoc < energy_img.cols - 1 && minVal == M[i - 1][minLoc + 1])
            minLoc++;
        seam->push(cv::Point(minLoc, i - 1));
    }
    return ret;
}

int find_horizontal_seam(cv::Mat const &energy_img, std::stack<cv::Point> *seam = NULL) {
    int M[energy_img.rows][energy_img.cols] = {0};

    // track forward using dp
    for (int i = 0; i < energy_img.rows; i++)
        M[i][0] = (energy_img.at<uchar>(i, 0) == UCHAR_MAX ? INT_MAX : energy_img.at<uchar>(i, 0));
    for (int j = 1; j < energy_img.cols; j++) {
        for (int i = 0; i < energy_img.rows; i++) {
            int min1 = i == 0 ? INT_MAX : std::min(M[i - 1][j - 1], M[i][j - 1]);
            int min2 = i == energy_img.rows - 1 ? INT_MAX : std::min(M[i][j - 1], M[i + 1][j - 1]);
            // The first case means this pixel is masked, the second means there isn't a path to this pixel.
            if(energy_img.at<uchar>(i, j) == UCHAR_MAX || std::min(min1, min2) == INT_MAX) 
                M[i][j] = INT_MAX;
            else 
                M[i][j] = energy_img.at<uchar>(i, j) + std::min(min1, min2);
        }
    }

    // backtrack to find the seam
    int minLoc = 0;
    int minVal = INT_MAX, ret;
    for (int i = 0; i < energy_img.rows; i++) {
        if(M[i][energy_img.cols - 1] < minVal) {
            minVal = M[i][energy_img.cols - 1];
            minLoc = i;
        }
    }
    ret = minVal;
    // The first case means not cal the seam, the second means can't find a seam.
    if(seam == NULL || minVal == INT_MAX) return 0;
    while(!seam->empty()) seam->pop();
    seam->push(cv::Point(energy_img.cols - 1, minLoc));
    for (int j = energy_img.cols - 1; j > 0; j--) {
        minVal -= energy_img.at<uchar>(minLoc, j);
        if(minLoc > 0 && minVal == M[minLoc - 1][j - 1])
            minLoc--;
        else if(minLoc < energy_img.cols - 1 && minVal == M[minLoc + 1][j - 1])
            minLoc++;
        seam->push(cv::Point(j - 1, minLoc));
    }
    return ret;
}

void find_optimal_seams_order(cv::Mat const &energy_img, cv::Size const &new_size, std::stack<char> &seams_order) {
    int c = energy_img.cols - new_size.width, r = energy_img.rows - new_size.height;
    long long T[r][c] = {0};
    cv::Mat energy_imgs[r][c];
    
    // track forward
    energy_imgs[0][0] = energy_img;
    for (int j = 1; j < c; j++) {
        std::stack<cv::Point> seam;
        T[0][j] = find_vertical_seam(energy_imgs[0][j - 1], &seam);
        energy_imgs[0][j] = energy_imgs[0][j - 1].clone();
        remove_seam<uchar>(energy_imgs[0][j], seam, 'v');
    }
    for (int i = 1; i < r; i++) {
        std::stack<cv::Point> seam;
        T[i][0] = find_horizontal_seam(energy_imgs[i - 1][0], &seam);
        energy_imgs[i][0] = energy_imgs[i - 1][0].clone();
        remove_seam<uchar>(energy_imgs[i][0], seam, 'h');
    }
    for (int i = 1; i < r; i++) {
        for (int j = 1; j < c; j++) {
            std::stack<cv::Point> seam1, seam2;
            int min1 = T[i][j - 1] + find_vertical_seam(energy_imgs[i][j - 1], &seam1);
            int min2 = T[i - 1][j] + find_horizontal_seam(energy_imgs[i - 1][j], &seam2);
            if(min1 < min2) {
                T[i][j] = min1;
                energy_imgs[i][j] = energy_imgs[i][j - 1].clone();
                remove_seam<uchar>(energy_imgs[i][j], seam1, 'v');
            }
            else {
                T[i][j] = min2;
                energy_imgs[i][j] = energy_imgs[i - 1][j].clone();
                remove_seam<uchar>(energy_imgs[i][j], seam2, 'h');
            }
        }
    }

    // track back
    while(!seams_order.empty()) seams_order.pop();
    while(!(r == 1 || c == 1)) {
        if (T[r - 1][c - 1] == T[r - 1][c - 2] + find_vertical_seam(energy_imgs[r - 1][c - 2], NULL)) {
            seams_order.push('v');
            c--;
        }
        else if (T[r - 1][c - 1] == T[r - 2][c - 1] + find_horizontal_seam(energy_imgs[r - 2][c - 1], NULL)){
            seams_order.push('h');
            r--;
        }
        else
            std::runtime_error("track back error.");
    }
    while(c--)
        seams_order.push('v');
    while(r--)
        seams_order.push('h');
}

double get_average_energy(cv::Mat const &energy_img) {
    return cv::mean(energy_img).val[0];
}

double get_average_energy(cv::Mat const &img, int fun_type) {
    cv::Mat energy_img;
    _get_energy_img(img, energy_img, fun_type);
    return get_average_energy(energy_img);
}

int find_k_seams(cv::Mat const &energy_img, std::queue<std::stack<cv::Point>> &seams, int k, char direction = 'v') {
    int ret = 0;
    cv::Mat _energy_img = energy_img.clone();
    while(!seams.empty()) seams.pop();
    while(k--) {
        std::stack<cv::Point> seam;
        if(direction == 'v')
            ret += find_vertical_seam(_energy_img, &seam);
        else if(direction == 'h')
            ret += find_horizontal_seam(_energy_img, &seam);
        else
            throw std::invalid_argument("unknown direction type.");
        
        if(seam.size() == 0)
            return ret;
        seams.push(seam);
        while(!seam.empty()) {
            _energy_img.at<uchar>(seam.top()) = UCHAR_MAX;
            seam.pop();
        }
    }
    return ret;
}