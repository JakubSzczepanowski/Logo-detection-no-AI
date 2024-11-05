#pragma once
#include "utils.hpp"
#include <list>

void analyze_colors(const Image& image);

cv::Mat segmentize_image(const Image& image, bool predicate(const HSV& hsv));

cv::Mat& erosion(cv::Mat& I);

cv::Mat& dilation(cv::Mat& I);

cv::Mat& open(cv::Mat& I);

cv::Mat& close(cv::Mat& I);

BoundingBox find_object(const cv::Mat& I);

std::pair<double, double> find_mass_center(cv::Mat& I);

std::list<Point> find_contours(const cv::Mat& I);

std::vector<BoundingBox> find_close_areas(const cv::Mat& I, std::list<Point> contours);