
#include <stdint.h>
#include <opencv2/core/hal/interface.h>
#include "utils.hpp"

Image::Image(int rows, int cols, HSV** image) : rows(rows), cols(cols), image(image) {}

void Image::Dispose() {
	for (int i = 0; i < rows; i++) {
		delete[] image[i];
	}

	delete[] image;
}

int Image::getRows() const {
	return rows;
}

int Image::getCols() const {
	return cols;
}

HSV** Image::getImage() const {
	return image;
}

Image::~Image() {
	Dispose();
}