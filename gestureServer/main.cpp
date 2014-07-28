#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <iostream>

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#define PI 3.14159265

using namespace cv;
using namespace std;


//declare the local functions
void morphologicalImgProc(Mat &frame);
string integerToString(int num);
int angleToCenter(const Point &v1, const Point &v2);
String doAction(int totalAngleOfFinger, int fingerSize);
void sendResult(String msg);


//calculate the angle between two points
int angleToCenter(const Point &finger, const Point &center) {
	float y_angle = center.y - finger.y; //center = 1;
	float x_angle = finger.x - center.x;// tip =2;
	float theta = atan(y_angle / x_angle);
	int angleFinger = (int)round(theta * 180 / PI);
	return angleFinger;
}

//convert the integer to string
string integerToString(int num) {
	stringstream strings;
	strings << num;
	string s = strings.str();
	return s;
}

//morphological Image processing
//Erosion -> dilation -> closing the frame ensure to get the better performance
void morphologicalImgProc(Mat &frame) {
	Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(9, 9), Point(5, 5));
	Mat element1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(7, 7), Point(5, 5));
	cv::dilate(frame, frame, element);
	cv::erode(frame, frame, element);
	cv::morphologyEx(frame, frame, MORPH_OPEN, element);
	cv::morphologyEx(frame, frame, MORPH_CLOSE, element);
}

//the important function to track the hand, the algorithm is described in the report
std::string trackHand(Mat src, Mat &dest) {
	//initialization local variables
	Rect boundRect;
	int largestObj;
	int boundingBoxHeight = 0;
	vector<vector<Point> > contours; //store all the contours
	vector<vector<Point> > contoursSet(contours.size());//store large contours
	vector<Vec4i> hierarchy;
	vector<Point> convexHullPoint;
	vector<Point> fingerPoint;
	Point centerP;
	int numObjects = 0;
	double area = 0;
	double maxArea = 0;
	bool handFound = false;
	String resultMsg = "no-command";
	//find all the contours in the threshold Frame
	findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	numObjects = hierarchy.size();
	for (unsigned int i = 0; i < contours.size(); i++) {
		Mat tempContour = Mat(contours[i]);
		area = contourArea(tempContour);
		if (area > maxArea) {
			maxArea = area;
			largestObj = i;
		}
	}
	if (maxArea > 4000){
		handFound = true;
		boundRect = boundingRect(contours[largestObj]);
		//draw the boundary of the object
		drawContours(dest, contours, largestObj, Scalar(0, 0, 255), 3, 8, hierarchy);
		//find the convex points for the largest object which is hand
		convexHull(contours[largestObj], convexHullPoint, true, true);
		approxPolyDP(Mat(contours[largestObj]), contours[largestObj], 3, true);
		//use moment method to find the center point
		Moments moment = moments(Mat(contours[largestObj]), true);
		int centerX = moment.m10 / moment.m00;
		int centerY = moment.m01 / moment.m00;
		Point centerPoint(centerX, centerY);
		centerP = centerPoint;
		Point printPoint(centerX, centerY + 15);
		Point printPoint1(boundRect.x, boundRect.y);
		circle(dest, centerPoint, 8, Scalar(255, 0, 0), CV_FILLED);
		//put the BoundingBox in the contour region
		rectangle(dest, boundRect, Scalar(0, 0, 255), 2, 8, 0);
		boundingBoxHeight = boundRect.height;
		//if( boundingBoxHeight <= 200)
		//	handFound = false;
		if (handFound) {
			int countHullPoint = convexHullPoint.size();
			int maxdist = 0;
			int pos = 0;
			for (int j = 1; j < countHullPoint; j++) {
				pos = j;
				if (centerP.y >= convexHullPoint[j].y && centerP.y >= convexHullPoint[pos].y) {
					pos = j;
					int dist = (centerP.x - convexHullPoint[j].x) ^ 2 + (centerP.y - convexHullPoint[j].y) ^ 2;
					if (abs(convexHullPoint[j - 1].x - convexHullPoint[j].x) < 12){
						if (dist > maxdist){
							maxdist = dist;
							pos = j;
						}
					}
					else if (j == 0 || abs(convexHullPoint[j - 1].x - convexHullPoint[j].x) >= 12){
						fingerPoint.push_back(convexHullPoint[pos]);
						cv::line(dest, centerP, convexHullPoint[pos], Scalar(0, 255, 0), 3, CV_AA, 0);
						circle(dest, convexHullPoint[pos], 8, Scalar(255, 0, 0), CV_FILLED);
						pos = j;
					}
				}
			}

			//get the size the fingers, and calculate the total angle of these fingers
			int countFinger = fingerPoint.size();
			int angle = 0;
			if (countFinger <= 5){
				for (int x = 0; x < countFinger; x++){
					angle = angle + abs(angleToCenter(fingerPoint[x], centerP));
				}
			}
			//cout << angle << endl;
			resultMsg = doAction(angle, countFinger);
			//sendResult(resultMsg);
			//putText(dest, integerToString(countFinger), printPoint, 1, 5, Scalar(0, 255, 0), 1, 5, false);
		}
	}
	return resultMsg;
}

////send out the result signal
void sendResult(String msg){
	cout << "Command: " << msg << endl;
}

//action performed based on the number of fingers and the total angle
//1.  5 fingers && total angle: 265 - 295
//2.  4 fingers && total angle: 235 - 260
//3.  3 fingers && total angle: 190 - 210
//4.  2 fingers && total angle: 110 - 130
//5.  1 finger && total angle:  65 - 85
String doAction(const int totalAngleOfFinger, const int fingerSize){
	String result = "no-command";
	if (totalAngleOfFinger >= 265 && totalAngleOfFinger <= 295 && (fingerSize == 5))
		result = "scroll-up";
	else if (totalAngleOfFinger >= 240 && totalAngleOfFinger <= 265 && fingerSize == 4)
		result = "scroll-down";
	else if (totalAngleOfFinger >= 190 && totalAngleOfFinger <= 210 && fingerSize == 3)
		result = "back-page";
	else if (totalAngleOfFinger >= 100 && totalAngleOfFinger <= 130 && (fingerSize == 2))
		result = "refresh-page";
	else if (totalAngleOfFinger >= 65 && totalAngleOfFinger <= 95 && (fingerSize == 1))
		result = "color-divs";
	return result;
}



typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::connection_hdl;

class count_server {
public:
	count_server() : m_count(0) {
		//setup server
		m_server.init_asio();
		m_server.set_open_handler(bind(&count_server::on_open, this, _1));
		m_server.set_close_handler(bind(&count_server::on_close, this, _1));

		//Setup Machine Vision Module
	}

	void on_open(connection_hdl hdl) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_connections.insert(hdl);
	}

	void on_close(connection_hdl hdl) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_connections.erase(hdl);
	} 

	void on_message(connection_hdl hdl, server::message_ptr msg) {
		for (auto it : m_connections) {
			m_server.send(it, msg);
		}
	}

	void count() {
		Mat cameraFrame, blurFrame, threshold1, threshold2, closedFrame, hsvFrame, colorObjectFrame, thresholdFrame;
		VideoCapture stream1;
		Mat grayscale;
		Mat fgMaskMOG;
		Mat foreground;
		Mat background;
		Ptr <BackgroundSubtractor> pMOG;
		pMOG = new BackgroundSubtractorMOG();

		//default the capture frame size to the certain size & open the camera
		stream1.set(CV_CAP_PROP_FRAME_WIDTH, 640);
		stream1.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
		stream1.open(0);
		background = imread("background.jpg");
		//report the error if the camera not connected properly
		if (!stream1.isOpened()) {
			cout << "cannot open camera";
		}

		int delay = 0;
		while (1) {	

			//get image from stream
			stream1.read(cameraFrame);
			pMOG->operator ()(cameraFrame, foreground);
			medianBlur(foreground, foreground, 3);
			cv::cvtColor(foreground, foreground, CV_GRAY2BGR);

			//switch the RGB to HSV space
			cv::cvtColor(foreground, hsvFrame, CV_BGR2HSV);

			//testing in the blue glove on hand
			//need to adjust before the live demo
			//cv::inRange(hsvFrame, Scalar(86, 72, 142), Scalar(117, 181, 256), thresholdFrame);

			//Tracking actual hand in restricted background
			cv::inRange(hsvFrame, Scalar(0, 0, 255), Scalar(256, 256, 256), thresholdFrame);

			//blur image to remove basic imperfections
			medianBlur(thresholdFrame, thresholdFrame, 5);

			//do the morphological image processing
			//closing the frame
			morphologicalImgProc(thresholdFrame);

			//track the hand, put the bounding box around the hand
			//calculate the center point of the hand
			string command = trackHand(thresholdFrame, cameraFrame);

			namedWindow("Hand_Gesture_Detection");
			imshow("Hand_Gesture_Detection", cameraFrame);
			namedWindow("foreground");
			imshow("foreground", foreground);

			if (waitKey(10) >= 0)
				break;
			//release the memory
			cameraFrame.release();

			delay++;
			if (delay > 10){
				delay = 0;
				std::cout << command << std::endl;
				std::lock_guard<std::mutex> lock(m_mutex);
				for (auto it : m_connections) {
					m_server.send(it, command, websocketpp::frame::opcode::text);
				}
			}
		}
	}

	void run(uint16_t port) {
		m_server.listen(port);
		m_server.start_accept();
		m_server.run();
	}
private:
	typedef std::set<connection_hdl, std::owner_less<connection_hdl>> con_list;

	int m_count;
	server m_server;
	con_list m_connections;
	std::mutex m_mutex;
};

int main() {
	count_server server;
	std::thread t(std::bind(&count_server::count, &server));
	server.run(9002);
}