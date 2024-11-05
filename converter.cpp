
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include "utils.hpp"


Image convertBGRtoHSV(cv::Mat image) {
	CV_Assert(image.depth() != sizeof(uchar));
	cv::Mat_<cv::Vec3b> _image = image;

	int rows = _image.rows;
	int cols = _image.cols;

	HSV** hsv_image = new HSV* [rows];
	int temp, i, j;

	for (i = 0; i < rows; i++) {
		hsv_image[i] = new HSV[cols];
	}
	
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++) {
			int blue = _image(i, j)[0];
			int green = _image(i, j)[1];
			int red = _image(i, j)[2];
			HSV hsv;
			int hue, saturation, value;
			
			temp = std::min(std::min(blue, green), red);
			value = std::max(std::max(blue, green), red);

			if (temp == value)
				hue = 0;
			else {
				if (red == value)
					hue = ((green - blue) * 60 / (value - temp));
				if (green == value)
					hue = 120 + ((green - blue) * 60 / (value - temp));
				if (blue == value)
					hue = 240 + ((green - blue) * 60 / (value - temp));
			}
			if (hue < 0)
				hue = hue + 360;

			if (value == 0)
				saturation = 0;
			else
				saturation = (value - temp) * 100 / value;

			value = (100 * value) / 255;

			hsv.hue = hue;
			hsv.saturation = saturation;
			hsv.value = value;
			hsv_image[i][j] = hsv;
		}
	

	return Image(rows, cols, hsv_image);

	
}