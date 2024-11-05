
#include <opencv2/core.hpp>
#include <iostream>
#include <utility>
#include "utils.hpp"
#include <opencv2/imgproc.hpp>
#include <list>
#include <set>
#include <opencv2/highgui.hpp>

void analyze_colors(const Image& image) {

	int rows = image.getRows();
	int cols = image.getCols();
	HSV** hsv = image.getImage();
	std::vector<short> reds_hue;
	std::vector<std::pair<uint8_t, uint8_t>> reds;
	std::vector<std::pair<uint8_t, uint8_t>> whites;
	std::vector<short> whites_hue;

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			ushort hue = hsv[i][j].hue;
			uint8_t saturation = hsv[i][j].saturation;
			uint8_t value = hsv[i][j].value;

			if (hue < 60) {
				reds_hue.push_back(hue);
				reds.push_back(std::pair<uint8_t, uint8_t>(saturation, value));
			}
			else if (hue > 300) {
				reds_hue.push_back(-(360 - hue));
				reds.push_back(std::pair<uint8_t, uint8_t>(saturation, value));
			}

			if (saturation < 20 && value > 80) {
				whites.push_back(std::pair<uint8_t, uint8_t>(saturation, value));
				whites_hue.push_back(hue);
			}

		}
	std::cout << "Czerwona barwa: " << *std::min_element(reds_hue.begin(), reds_hue.end()) << " " << *std::max_element(reds_hue.begin(), reds_hue.end()) << std::endl;
	auto saturation_min_red = (*std::min_element(reds.begin(), reds.end(), [](auto a, auto b) {return a.first < b.first; })).first;
	auto saturation_max_red = (*std::max_element(reds.begin(), reds.end(), [](auto a, auto b) {return a.first < b.first; })).first;
	auto value_min_red = (*std::min_element(reds.begin(), reds.end(), [](auto a, auto b) {return a.second < b.second; })).second;
	auto value_max_red = (*std::max_element(reds.begin(), reds.end(), [](auto a, auto b) {return a.second < b.second; })).second;

	std::cout << "Czerwona barwa - saturation: " << (short)saturation_min_red << " " << (short)saturation_max_red << std::endl;
	std::cout << "Czerwona barwa - value: " << (short)value_min_red << " " << (short)value_max_red << std::endl;

	auto saturation_min = (*std::min_element(whites.begin(), whites.end(), [](auto a, auto b) {return a.first < b.first; })).first;
	auto saturation_max = (*std::max_element(whites.begin(), whites.end(), [](auto a, auto b) {return a.first < b.first; })).first;
	auto value_min = (*std::min_element(whites.begin(), whites.end(), [](auto a, auto b) {return a.second < b.second; })).second;
	auto value_max = (*std::max_element(whites.begin(), whites.end(), [](auto a, auto b) {return a.second < b.second; })).second;
	std::cout << "Bia³a barwa: " << *std::min_element(whites_hue.begin(), whites_hue.end()) << " " << *std::max_element(whites_hue.begin(), whites_hue.end()) << std::endl;
	std::cout << "Bia³a barwa - saturation: " << (short)saturation_min << " " << (short)saturation_max << std::endl;
	std::cout << "Bia³a barwa - value: " << (short)value_min << " " << (short)value_max << std::endl;
}

cv::Mat segmentize_image(const Image& image, bool predicate(const HSV& hsv)) {

	int rows = image.getRows();
	int cols = image.getCols();
	HSV** hsv = image.getImage();
	cv::Mat bin_image(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {

			if (predicate(hsv[i][j]))
				bin_image.at<uchar>(i, j) = 255;
			else
				bin_image.at<uchar>(i, j) = 0;
		}

	return bin_image;
}

cv::Mat& erosion(cv::Mat& I) {
	cv::Mat_<uchar> _I = I;
	cv::Mat_<uchar> _I_copy = _I.clone();


	for (int i = 1; i < I.rows - 1; ++i)
		for (int j = 1; j < I.cols - 1; ++j) {

			bool all_white = true;
			if (_I_copy(i, j) == 255) {
				for (char v = -1; v <= 1; v++) {
					for (char h = -1; h <= 1; h++) {
						if (_I_copy(i + v, j + h) == 0) {
							all_white = false;
							break;
						}
					}
					if (!all_white) break;
				}
				_I(i, j) = all_white ? 255 : 0;
			}
		}


	return I;
}

cv::Mat& dilation(cv::Mat& I) {
	cv::Mat_<uchar> _I = I;
	cv::Mat_<uchar> _I_copy = _I.clone();


	for (int i = 1; i < I.rows - 1; ++i)
		for (int j = 1; j < I.cols - 1; ++j) {

			if (_I_copy(i, j) == 255) {
				for (char v = -1; v <= 1; v++) {
					for (char h = -1; h <= 1; h++) {
						_I(i + v, j + h) = 255;
					}
				}
			}
		}

	return I;
}

cv::Mat& open(cv::Mat& I) {
	I = erosion(I);
	I = dilation(I);

	return I;
}

cv::Mat& close(cv::Mat& I) {
	I = dilation(I);
	I = erosion(I);

	return I;
}

BoundingBox find_object(const cv::Mat& I) {

	cv::Mat_<uchar> _I = I;
	BoundingBox box = BoundingBox(I.cols, I.rows);

	for (int i = 0; i < I.rows; i++)
		for (int j = 0; j < I.cols; j++) {
			if (_I(i, j) == 255) {
				box.x0 = std::min(box.x0, j);
				box.y0 = std::min(box.y0, i);
				box.x1 = std::max(box.x1, j);
				box.y1 = std::max(box.y1, i);
			}
		}
	if (box.x0 != 0) box.x0--;
	if (box.y0 != 0) box.y0--;

	if (box.x1 + 2 < I.cols) box.x1 += 2;
	if (box.y1 + 2 < I.rows) box.y1 += 2;

	return box;
}



std::pair<double, double> find_mass_center(cv::Mat& I) {

	cv::Mat_<uchar> _I = I;
	double x = 0.0;
	double y = 0.0;
	int count = 0;

	for (int i = 0; i < I.rows; i++)
		for (int j = 0; j < I.cols; j++) {
			if (_I(i, j) != 255) {
				x += j;
				y += i;
				count++;
			}
		}

	x /= count;
	y /= count;

	return std::make_pair(x, y);
}

std::list<Point> find_contours(const cv::Mat& I) {

	std::list<Point> contours;
	cv::Mat_<uchar> _I = I;

	for (int i = 1; i < I.rows - 1; i++)
		for (int j = 1; j < I.cols - 1; j++) {
			if (_I(i, j) == 255 && (_I(i, j) != _I(i - 1, j - 1) ||
				_I(i, j) != _I(i - 1, j) ||
				_I(i, j) != _I(i - 1, j + 1) ||
				_I(i, j) != _I(i, j + 1) ||
				_I(i, j) != _I(i - 1, j) ||
				_I(i, j) != _I(i + 1, j + 1) ||
				_I(i, j) != _I(i + 1, j) ||
				_I(i, j) != _I(i + 1, j - 1)))
				contours.push_back(Point(j, i));
		}

	return contours;
}

bool isContour(const cv::Mat& I, int i, int j) {

	cv::Mat_<uchar> _I = I;

	return (_I(i, j) == 255 &&
		(_I(i, j) != _I(i - 1, j - 1) ||
			_I(i, j) != _I(i - 1, j) ||
			_I(i, j) != _I(i - 1, j + 1) ||
			_I(i, j) != _I(i, j + 1) ||
			_I(i, j) != _I(i - 1, j) ||
			_I(i, j) != _I(i + 1, j + 1) ||
			_I(i, j) != _I(i + 1, j) ||
			_I(i, j) != _I(i + 1, j - 1)));
}

std::vector<BoundingBox> find_close_areas(const cv::Mat& I, std::list<Point> contours) {

	cv::Mat_<uchar> _I = I;
	std::vector<BoundingBox> close_areas;
	std::set<Point> prev_points;

	while (!contours.empty()) {

		Point starting_point = contours.front();
		Point current_point = starting_point;
		bool break_loop = false;
		BoundingBox box = BoundingBox(I.cols, I.rows);
		Point diagonal_point;
		
		int stringLength = 0;

		for (char v = -1; v <= 1; v++) {
			for (char h = -1; h <= 1; h++) {

				if (v != 0 || h != 0) {
					int i = starting_point.y + v;
					int j = starting_point.x + h;

					if (j < 1 || j >= I.cols - 1 || i < 1 || i >= I.rows - 1) {
						contours.remove(Point(j, i));
						prev_points.insert(starting_point);
						prev_points.insert(Point(j, i));
						contours.remove(starting_point);
						break;
					}

					if (i != starting_point.y && j != starting_point.x && prev_points.find(Point(j, i)) == prev_points.end() && isContour(I, i, j)) {
						diagonal_point = Point(j, i);
					}
					else if (prev_points.find(Point(j, i)) == prev_points.end() &&
						isContour(I, i, j)) {
						current_point = Point(j, i);
						prev_points.insert(starting_point);
						break_loop = true;
						box.x0 = std::min(box.x0, j);
						box.y0 = std::min(box.y0, i);
						box.x1 = std::max(box.x1, j);
						box.y1 = std::max(box.y1, i);
						contours.remove(current_point);
						stringLength++;
						break;
					}
				}
			}
			if (break_loop)
				break;
		}

		if (!break_loop && diagonal_point != Point()) {
			current_point = diagonal_point;
			prev_points.insert(starting_point);
			box.x0 = std::min(box.x0, diagonal_point.x);
			box.y0 = std::min(box.y0, diagonal_point.y);
			box.x1 = std::max(box.x1, diagonal_point.x);
			box.y1 = std::max(box.y1, diagonal_point.y);
			contours.remove(current_point);
			stringLength++;
		}

		bool searchForStartingPoint = true;

		while (searchForStartingPoint) {

			// WIZUALIZACJA
			/*if (stringLength % 50 == 0) {
				cv::Mat contours_image = cv::Mat::zeros(cv::Size(I.cols, I.rows), CV_8UC1);

				for (Point p : contours) {
					contours_image.at<uchar>(p.y, p.x) = 255;
				}
				cv::imshow("image", contours_image);
				cv::waitKey(-1);
			}*/

			diagonal_point = Point();
			break_loop = false;

			for (char v = -1; v <= 1; v++) {
				for (char h = -1; h <= 1; h++) {

					if (v != 0 || h != 0) {
						int i = current_point.y + v;
						int j = current_point.x + h;

						if (j < 1 || j >= I.cols - 1 || i < 1 || i >= I.rows - 1) {
							prev_points.insert(current_point);
							prev_points.insert(Point(j, i));
							contours.remove(Point(j, i));
							contours.remove(current_point);
							break;
						}

						if (stringLength > 50 && (starting_point.x >= current_point.x - 1 &&
							starting_point.x <= current_point.x + 1 &&
							starting_point.y >= current_point.y - 1 &&
							starting_point.y <= current_point.y + 1)) {

							searchForStartingPoint = false;
							break_loop = true;
							break;
						}

						if (i != current_point.y && j != current_point.x && prev_points.find(Point(j, i)) == prev_points.end() && isContour(I, i, j)) {
							diagonal_point = Point(j, i);
						}
						else if (prev_points.find(Point(j, i)) == prev_points.end() &&
							isContour(I, i, j)) {
							prev_points.insert(current_point);
							current_point = Point(j, i);
							break_loop = true;
							box.x0 = std::min(box.x0, j);
							box.y0 = std::min(box.y0, i);
							box.x1 = std::max(box.x1, j);
							box.y1 = std::max(box.y1, i);
							contours.remove(current_point);
							stringLength++;
							break;
						}
					}
				}
				if (break_loop)
					break;
			}
			if (!break_loop && diagonal_point != Point()) {
				current_point = diagonal_point;
				prev_points.insert(starting_point);
				box.x0 = std::min(box.x0, diagonal_point.x);
				box.y0 = std::min(box.y0, diagonal_point.y);
				box.x1 = std::max(box.x1, diagonal_point.x);
				box.y1 = std::max(box.y1, diagonal_point.y);
				contours.remove(current_point);
				stringLength++;
			}
			else if (!break_loop) {
				if (stringLength > 50 && starting_point.x >= current_point.x - 2 && starting_point.x <= current_point.x + 2 && starting_point.y >= current_point.y - 2 && starting_point.y <= current_point.y + 2)
					searchForStartingPoint = false;
				break;
			}
		}

		prev_points.insert(starting_point);
		contours.remove(starting_point);

		if (searchForStartingPoint == false) {
			if (box.x0 != 0) box.x0--;
			if (box.y0 != 0) box.y0--;

			if (box.x1 + 2 < I.cols) box.x1 += 2;
			if (box.y1 + 2 < I.rows) box.y1 += 2;

			close_areas.push_back(box);
		}
	}

	return close_areas;
}