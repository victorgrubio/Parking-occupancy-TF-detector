/* C++ code from automatic parking spot detection

We get parking lines from image of parking lot and get rid of noise: 
*/

Canny(inputImage, helpMatrix, 450, 400, 3);
cvtColor(helpMatrix, helpMatrix2, CV_GRAY2BGR);
vector<Vec4i> lines;
HoughLinesP(helpMatrix, lines, 1, CV_PI / 180, 7, 10, 10);
for (size_t i = 0; i < lines.size(); i++)
{
    Vec4i l = lines[i];
    line(helpMatrix2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 5, CV_AA)
}

/*We use double dilate and substract their results to get mask of lines:
*/

Mat element2 = getStructuringElement(CV_SHAPE_RECT, Size(3, 3));
cv::erode(helpMatrix2, helpMatrix2, element);
cv::dilate(helpMatrix2, helpMatrix2, element2);

morphologyEx(helpMatrix2, mark, CV_MOP_DILATE, element,Point(-1,-1), 3);
morphologyEx(helpMatrix2, mark2, CV_MOP_DILATE, element, Point(-1, -1), 2);
result = mark - mark2;

/*We use Canny and Hough lines, this time for removing the connecting line between each parking spot:*/

Canny(resu, mark, 750, 800, 3);
cvtColor(mark, mark2, CV_GRAY2BGR);
mark2 = Scalar::all(0);
vector<Vec4i> lines3;
HoughLinesP(mark, lines3, 1, CV_PI / 180, 20, 15, 10);
for (size_t i = 0; i < lines3.size(); i++)
{
    Vec4i l = lines3[i];
    line(mark2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, CV_AA);
}

/*We use this as a mask for finding contours for the Watershed algorithm and get result with detected parking spots, each colored with different color:
 */

vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
findContours(markerMask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
int contourID = 0;
for (; contourID >= 0; contourID = hierarchy[contourID][0], parkingSpaceCount++)
{
    drawContours(markers, contours, contourID, Scalar::all(parkingSpaceCount + 1), -1, 8, hierarchy, INT_MAX);
}
watershed(helpMatrix2, markers);
Mat wshed(markers.size(), CV_8UC3);
for (i = 0; i < markers.rows; i++)
    for (j = 0; j < markers.cols; j++)
    {
        int index = markers.at<int>(i, j);
        if (index == -1)
            wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
        else if (index <= 0 || index > parkingSpaceCount)
            wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        else
            wshed.at<Vec3b>(i, j) = colorTab[index - 1];
    }
/*
If our user is not satisfied with this result, he can always draw the seeds for watershed himself, or just adjust these seeds (img is the name of matrix, where user can see markers and markerMask matrix, where seeds are stored):
*/
Point prevPt(-1, -1);
static void onMouse(int event, int x, int y, int flags, void*)
{
    if (event == EVENT_LBUTTONDOWN) prevPt = Point(x, y);
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
    {
        Point pt(x, y);
        if (prevPt.x < 0)
            prevPt = pt;
        line(markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0);
        line(img, prevPt, pt, Scalar::all(255), 5, 8, 0);
        prevPt = pt;
        imshow("image", img);
    }
}

/* We have our spots stored, so we know their exact location, now its time to determine, wheter, or not check the lot again, if some vehicles are moving. For this purpose we need to detect movement on the lot with backgroundSubstraction, which can constantly learn what is static in image:
*/
Ptr<BackgroundSubtractor> pMOG2;
pMOG2 = new BackgroundSubtractorMOG2(3000, 20.7,true);
# We will give the MOG every frame captured from video feed and see what it results:
pMOG2->operator()(frame, matMaskMog2,0.0035);
imshow("MOG2", matMaskMog2);

/*As we can see, there is some noise detected – this noise represents for example moving leaves on trees, so it is necessary to remove it: */

cv::morphologyEx(matMaskMog2, matMaskMog2, CV_MOP_ERODE, element);
cv::medianBlur(matMaskMog2, matMaskMog2, 3);
cv::morphologyEx(matMaskMog2, matMaskMog2, CV_MOP_DILATE, element2);

/* Finally we find coordinates of moving object from MOG and draw a rectangle with random color around it (result can be seen at the top): */


scv::findContours(matMaskMog2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
vector<vector<Point> > contours_poly(contours.size());
vector<Rect> boundRect(contours.size());
 
for (int i = 0; i < contours.size(); i++)
{
approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
    boundRect[i] = boundingRect(Mat(contours_poly[i]));     
}
RNG rng(01);
for (int i = 0; i< contours.size(); i++)
{
Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)); 
    rectangle(frame, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
}


