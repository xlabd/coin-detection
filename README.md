# coin-detection
Detect and display coins using OpenCV functions in C++.

This application reads the input image, counts the number of coins, and annotates the detected coins. To preprocess the images, we employ thresholding and morphology operations viz. dilation and erosion. To detect the coins, we use OpenCV SimpleBlobDetector and contour analysis. Finally, we visualise it using connected component analysis and annotating the images.

This project is a part of OpenCV's Introduction to Computer Vision.

## Execute the code
To execute the code, follow these steps:

1. Set the `OpenCV_DIR` in `CMakeLists.txt`.
2. Clean the build directory. `rm -rf build`.
3. Configure the project. `cmake -B build -S .`.
4. Build the code. `make -C build`.
5. Run code. `./build/detect-coins`.

