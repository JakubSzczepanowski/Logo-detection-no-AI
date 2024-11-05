
#pragma once

struct HSV {
	ushort hue;
	uint8_t saturation;
	uint8_t value;
};

struct BoundingBox {
	int x0, y0, x1, y1;

	BoundingBox() : x0(0), y0(0), x1(0), y1(0) {}
	BoundingBox(int width, int height) : x0(width), y0(height), x1(0), y1(0) {}
	BoundingBox(int x0, int y0, int x1, int y1) : x0(x0), y0(y0), x1(x1), y1(y1) {}
};

struct Point {
	int x, y;

	Point() : x(0), y(0) {}
	Point(int x, int y) : x(x), y(y) {}

	friend bool operator==(const Point& left, const Point& right) {
		return (left.x == right.x) && (left.y == right.y);
	}

	friend bool operator!=(const Point& left, const Point& right) {
		return !(left == right);
	}

	friend bool operator<(const Point& left, const Point& right) {
		if (left.x == right.x) {
			return left.y < right.y;
		}
		return left.x < right.x;
	}

	friend bool operator>(const Point& left, const Point& right) {
		if (left.x == right.x) {
			return left.y > right.y;
		}
		return left.x > right.x;
	}
};

class Image {
private:
	int rows, cols;
	HSV** image;

public:
	Image(int rows, int cols, HSV** image);
	int getRows() const;
	int getCols() const;
	HSV** getImage() const;
	void Dispose();
	~Image();
};