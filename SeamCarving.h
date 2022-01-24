#ifndef SEAMCARVER_H_
#define SEAMCARVER_H_

#include <stack>
#include <queue>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>

const int ENERGY_FUN_SOBEL_L1 = 0x00;
const int ENERGY_FUN_SOBEL_L2 = 0x01;
/*
const int ENERGY_FUN_HOG = 0x10
const int ENERGY_FUN_ENTROPY = 0x20
const int ENERGY_FUN_SEGMENTATION = 0x30
*/

/**
 * @brief Calculate energy img of the original img without normalize.
 * @param img Image to be processed.
 * @param energy_img The output energy image.
 * @param fun_type Image energy function to be used. Soebl-L1, Sobel-L2 are supported.
 */
void _get_energy_img(cv::Mat const &img, cv::Mat &energy_img, int fun_type);

/**
 * @brief Calculate energy img of the original img with normalize.
 * @param img Image to be processed.
 * @param energy_img The output energy image.
 * @param fun_type Image energy function to be used. Soebl-L1, Sobel-L2 are supported.
 */
void get_energy_img(cv::Mat const &img, cv::Mat &energy_img, int fun_type);

/**
 * @brief Find one vertical seam with lowest s* in energy img.
 * @param energy_img The energy image.
 * @param seam The points of the seam, where y increases strictly as pop.
 * @return The total energy of the seam.
 */
int find_vertical_seam(cv::Mat const &energy_img, std::stack<cv::Point> *seam);

/**
 * @brief Find one horizontal seam with lowest s* in energy img. 
 * @note In order to reduce the time cost of rot90° operation, we won't reuse the code in function find_vertical_seam.
 * @param energy_img The energy image.
 * @param seam The points of the seam, where x increases strictly as pop.
 * @return The total energy of the seam.
 */
int find_horizontal_seam(cv::Mat const &energy_img, std::stack<cv::Point> *seam);

/**
 * @brief Find optimal seams order between vertical seam operation and horizontal seam operation using dp.
 * @param energy_img Energy img of the original image.
 * @param new_size The new size.
 * @param seams_order The optimal order, where 'v' presents vertical seam and 'h' present horizontal seam.
 */
void find_optimal_seams_order(cv::Mat const &energy_img, cv::Size const &new_size, std::stack<char> &seams_order);

/**
 * @brief Get average of img form energy img.
 * @param energy_img The corresponding energy img to the original img.
 * @return The average energy of original img.
 */
double get_average_energy(cv::Mat const &energy_img);

/** @overload */
double get_average_energy(cv::Mat const &img, int fun_type);

/**
 * @brief Find k seams with lowest s* in energy img
 * @param energy_img The energy image.
 * @param seams The vector of seams, and each seam cotains a stack of points, where y increases strictly as pop.
 * @param k The number of seams, k must be less than half of the width of the energy_img.
 * @return The total energy of the seams. -1 if k > energy_img.cols / 2.
 * @param direction The direction of seam. Use 'v' means vertical or 'h' means horizontal.
 */
int find_k_seams(cv::Mat const &energy_img, std::queue<std::stack<cv::Point>> &seams, int k, char direction);


#include "SeamCarving_impl.h"

#endif // SEAMCARVER_H_