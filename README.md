# NanoTrack_Tensorrt_Cpp

This  repository deployed **NanoTrack**  (developed by **@ HonglinChu** ) in **C++** and used **Tensorrt** for inference acceleration.

OpenCV 4.8.0 have the SKD for NanoTrack, you only need to load onnx files and write  your code like this:

```C++
#include <opencv2/core/utility.hpp>  
#include <opencv2/tracking/tracking.hpp>  
#include <opencv2/videoio.hpp>  
#include <opencv2/highgui.hpp>  
#include <iostream>  
#include <chrono>  
  
#include <cstring>  
using namespace std;  
using namespace cv;  
  
int main( int argc, char** argv ){  
    Mat frame;  
    Ptr<Tracker> tracker;  
  
    bool is_nano = false;  
  
    ////////////////////////////////////////////    
    /// nano track need onnx files  
    /////this part is to import onnx files of nano track.    
    cv::TrackerNano::Params params;  
    params.backbone = cv::samples::findFile("opencv_nano_track_model/nano_track/nanotrack_backbone_sim.onnx");  
    params.neckhead = cv::samples::findFile("opencv_nano_track_model/nano_track/nanotrack_head_sim.onnx");  
    tracker = TrackerNano::create(params);  
    is_nano = true;  
    /////  

    VideoCapture cap(0);  
  
    cap >> frame;  
  
    //resize(frame,frame,Size(960,540));  
    auto roi = selectROI(frame, false);  
  
    //quit if ROI was not selected  
    if(roi.width==0 || roi.height==0)  
        return 0;  
  
    // initialize the tracker  
    tracker->init(frame,roi);  
    // perform the tracking process  
    printf("Start the tracking process, press ESC to quit.\n");  
    for ( ;; ){  
        // get frame from the video  
        cap >> frame;  
        //resize(frame,frame,Size(960,540));  
        if(frame.rows==0 || frame.cols==0)  
            break;  
        // update the tracking result  
        // cal fps        auto begin = std::chrono::high_resolution_clock::now();  
        tracker->update(frame, roi);  
        auto end = std::chrono::high_resolution_clock::now();  
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);  
        int fps = 1000/elapsed.count();  
        string  fps_s = to_string(fps);  
        // draw the tracked object  
        rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );  
        cv::putText(frame, "FPS:"+fps_s, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1, cv::LINE_8, false);  
        // show image with the tracked object  
        imshow("tracker",frame);  
        cv::waitKey(2);  
        if(is_nano){  
            cv::waitKey(20);  
        }  
        //quit on ESC button  
        if(waitKey(1)==27)break;  
    }  
    return 0;  
}
```
**The inference of NanoTrack is totally by *CPU* when you use OpenCV SDK .** It can achieve ***40fps*** on my computer.
Considering that it only uses CPU for inference, it will inevitably consume more CPU resources, which is not conducive to deployment in some resource limited devices.
What's more, only OpenCV beyond 4.8.0 have method of NanoTrack. This imposes restrictions on the version of OpenCV for your project

**Some  devices(like Jetson) can use GPU for acceleration. So this repository choose to show how to deploy NanoTrack using C++ and Tensorrt.**

## 1. Requirement
1. OpenCV
2. Tensorrt ( I use 8.5.2.2. There should be no restrictions on its version but I have not  tested yet)
3. CUDA (I use 11.8)

**NOTE**: There is some correspondence between Tensorrt's version and CUDA's version. You can check the introduction on Nvidia's official website.

## 2. RUN
Choose a path to open the terminal, The following commands are all executed in the terminal.  

1. ``git clone https://github.com/ZhangLi1210/NanoTrack_Tensorrt_Cpp.git``
2. ``cd NanoTrack_Tensorrt_Cpp``
3. ``sudo chmod 777 create_trt_engine.sh``

4.  ``sudo vi create_trt_engine.sh`` change the trtexec path to yours . save and back to the terminal. **if you use Jetson with tensorrt , your  trtexec path normally in /usr/src/tensorrt/bin/trtexec**
5. Cheate engine files from onnx files by using ``./create_trt_engine.sh`` The engine files will be saved in engine folder.
6. Edit cmakelist.  change the OpenCV and Tensorrt path to yours.
7. ``cd build``
8. ``cmake ..``
9. ``make``

After all of above,  Run ``./nano_track_trt`` .The program will attempt to obtain the camera on your computer and track the area you have selected using the mouse.

**On my computer, the maximum frame rate can reach *250* fps! And the result is similar to using NanoTrack in OpenCV SDK**
