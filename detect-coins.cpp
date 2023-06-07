// Coin Detection Assignment
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


void displayImage(Mat image){
	imshow("Image", image);
	waitKey(0);
	destroyAllWindows();
}

Mat displayConnectedComponents(Mat &im)
{
	// Make a copy of the image
	Mat imLabels = im.clone();

	// First let's find the min and max values in imLabels
	Point minLoc, maxLoc;
	double min, max;

	// The following line finds the min and max pixel values
	// and their locations in an image.
	minMaxLoc(imLabels, &min, &max, &minLoc, &maxLoc);

	// Normalize the image so the min value is 0 and max value is 255.
	imLabels = 255 * (imLabels - min) / (max - min);

	// Convert image to 8-bits
	imLabels.convertTo(imLabels, CV_8U);

	// Apply a color map
	Mat imColorMap;
	applyColorMap(imLabels, imColorMap, COLORMAP_JET);

	return imColorMap;
}

int main(){
	
	// Image path
	string imagePath = "images/CoinsA.png";
	// Read image
	// Store it in the variable image
	Mat image = imread(imagePath, 1);

	Mat imageCopy = image.clone();
	
	displayImage(image);
	
	// Convert image to grayscale
	// Store it in the variable imageGray
	Mat imageGray;
	cvtColor(image, imageGray, COLOR_BGR2GRAY);
	
	displayImage(imageGray);
	
	// Split cell into channels
	// Store them in variables imageB, imageG, imageR
	Mat imageB, imageG, imageR, channels[3];
	split(image, channels);
	imageB = channels[0];
	imageG = channels[1];
	imageR = channels[2];
	
	displayImage(imageB);
	displayImage(imageG);
	displayImage(imageR);
	
	// Perform thresholding
	Mat dst;
	threshold(imageG, dst, 50, 255, THRESH_BINARY_INV);
	
	// Modify as required
	displayImage(dst);
	
	// Perform morphological operations
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3)), imageEroded;
	erode(dst, imageEroded, kernel);
	
	displayImage(imageEroded);
	
	// Get structuring element/kernel which will be used for dilation
	Mat imageDilated;
	kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	dilate(imageEroded, imageDilated, kernel, Point(-1,-1), 2);
	
	displayImage(imageDilated);
	
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	params.blobColor = 0;

	params.minDistBetweenBlobs = 2;

	// Filter by Area
	params.filterByArea = false;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.8;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.8;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.8;
	
	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	
	// Detect blobs
	vector<KeyPoint> keypoints;
	detector->detect(imageDilated, keypoints);
	
	// Print number of coins detected
	cout << keypoints.size();
	
	// Mark coins using image annotation concepts we have studied so far
	int x,y;
	int radius;
	double diameter;
	for(int i = 0; i < keypoints.size(); i++){
	    KeyPoint j = keypoints[i];
	    Point center = j.pt;
	    x=(int)center.x;
	    y=(int)center.y;
	    circle(image, Point(x, y), 5, Scalar(255, 255, 255), -1);
	    circle(image, Point(x, y), ((double)j.size/2.0), Scalar(255, 255, 255), 3);
	}
	
	displayImage(image);
	
	// Find connected components
	// Use displayConnectedComponents function provided above
	// reverse threshold
	threshold(imageDilated, imageDilated, 1, 255, THRESH_BINARY_INV);
	Mat labels;
	// find connected components
	int num_coins = connectedComponents(imageDilated, labels);
	Mat colorMap = displayConnectedComponents(labels);
	cout << "Number of components detected = " << num_coins;
	
	displayImage(colorMap);
	
	// Find all contours in the image
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imageDilated, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	
	// Print the number of contours found
	cout << "Number of contours found = " << contours.size();
	
	// Draw all contours
	Mat imageContour = imageCopy.clone();
	drawContours(imageContour, contours, -1, Scalar(0,0,0), 3);
	displayImage(imageContour);
	
	// Remove the inner contours
	// Display the result
	image = imageCopy.clone();
	drawContours(image, contours, 0, Scalar(0,0,0), 25, LINE_AA, hierarchy, 0);
	
	displayImage(image);
	
	// Print area and perimeter of all contours
	double a, p, min_area = 999;
	for (int i = 0; i < contours.size(); i++){
	    a = contourArea(contours[i]);
	    p = arcLength(contours[i], true);
	    cout << "Contour #" << i << " has area = " << a << ",\t and perimeter = " << p << endl;
	    if(a < min_area)
	        min_area = a;
	}
	
	// Print maximum area of contour
	// This will be the box that we want to remove
	cout << "Minimum area of contour = " << min_area << endl;
	
	// Remove this contour and plot others
	image = imageCopy.clone();
	for(int i = 1; i < contours.size(); i++){
	    drawContours(image, contours, i, Scalar(0,0,0), 5);
	}
	
	// Fit circles on coins
	image = imageCopy.clone();
	Point2f center;
	float coin_radius;
	for (size_t i=0; i < contours.size(); i++){
	    minEnclosingCircle(contours[i], center, coin_radius);
	    circle(image, center, coin_radius, Scalar(255, 255, 255), 5);
	    circle(image, center, 2, Scalar(255, 255, 255), -1);
	}
	cout << "Number of coins detected = " << contours.size();
	
	displayImage(image);
	
	// Image path
	imagePath = "images/CoinsB.png";
	// Read image
	// Store it in variable image
	image = imread(imagePath, 1);
	resize(image, image, Size(), 0.25, 0.25);
	imageCopy = image.clone();
	
	// Convert image to grayscale
	// Store it in the variable imageGray
	cvtColor(image, imageGray, COLOR_BGR2GRAY);
	
	displayImage(imageGray);
	
	// Split cell into channels
	// Store them in variables imageB, imageG, imageR
	split(image, channels);
	imageB = channels[0];
	imageG = channels[1];
	imageR = channels[2];

	
	displayImage(imageB);
	displayImage(imageG);
	displayImage(imageR);
	
	// Perform thresholding
	threshold(imageB, dst, 120, 255, THRESH_BINARY_INV);
	
	displayImage(dst);
	
	// Perform morphological operations
	cout << "First we'll remove the small white points in the image" << endl;
	kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	erode(dst, imageEroded, kernel, Point(-1,-1), 2);

	cout << "Now we'll increase the white area which was reduced in the earlier step." << endl;
	cout << "We'll double the prevous amount to compensate." << endl;
	kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	dilate(imageEroded, imageDilated, kernel, Point(-1,-1), 4);

	displayImage(imageDilated);

	// finding the outer contours only
	findContours(imageDilated, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// drawing the (outer) contours, but filling it completely
	drawContours(imageDilated, contours, -1, Scalar(255), -1);
	cout << "We have drawn the contours on the image, but filled it completeley" << endl;
	cout << "This is the image which will be used from here onwards" << endl;

	displayImage(imageDilated);
	
	// Setup SimpleBlobDetector parameters.
	cout << "Changing the blob colour to white" << endl;
	params.blobColor = 255;

	params.minDistBetweenBlobs = 2;

	// Filter by Area
	params.filterByArea = false;

	// Filter by Circularity
	cout << "Decreasing the circularity value to detect all the blobs" << endl;
	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.6;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.8;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.8;
	
	// Set up detector with params
	detector = SimpleBlobDetector::create(params);
	
	// Detect blobs
	detector->detect(imageDilated, keypoints);
	
	// Print number of coins detected
	cout << keypoints.size();
	
	// Mark coins using image annotation concepts we have studied so far
	for(int i = 0; i < keypoints.size(); i++){
	    KeyPoint j = keypoints[i];
	    Point center = j.pt;
	    x=(int)center.x;
	    y=(int)center.y;
	    circle(image, Point(x, y), 5, Scalar(255, 255, 255), -1);
	    circle(image, Point(x, y), ((double)j.size/2.0), Scalar(255, 255, 255), 15);
	}
	displayImage(image);

	// Find connected components
	num_coins = connectedComponents(imageDilated, labels);
	colorMap = displayConnectedComponents(labels);
	cout << "Number of components detected = " << num_coins;
	displayImage(colorMap);

	// Find all contours in the image
	findContours(imageDilated, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	
	// Print the number of contours found
	cout << "Number of contours found = " << contours.size() << endl;
	
	// Draw all contours
	image = imageCopy.clone();
	drawContours(image, contours, -1, Scalar(0,0,0), 25, LINE_AA);
	displayImage(image);
	
	// Print area and perimeter of all contours
	min_area = 999;
	for (int i = 0; i < contours.size(); i++){
	    a = contourArea(contours[i]);
	    p = arcLength(contours[i], true);
	    cout << "Contour #" << i+1 << " has area = " << a << ",\t and perimeter = " << p << endl;
	    if(a < min_area)
	        min_area = a;
	}
	
	// Fit circles on coins
	image = imageCopy.clone();
	for (size_t i=0; i < contours.size(); i++){
	    minEnclosingCircle(contours[i], center, coin_radius);
	    circle(image, center, coin_radius, Scalar(255, 255, 255), 5);
	    circle(image, center, 2, Scalar(255, 255, 255), -1);
	}
	cout << "Number of coins detected = " << contours.size();
	
	displayImage(image);
	
	return 0;
}