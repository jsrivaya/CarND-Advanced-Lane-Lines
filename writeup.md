
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/Undistorted.png "Undistorted"
[image2]: ./writeup_images/OriginalvsUndistored.png "Raw Correction"
[image3]: ./writeup_images/Gradients.png "Gradients"
[image4]: ./writeup_images/HLSvsBinary.png "HLS channels vs HLS Binary channels"
[image5]: ./writeup_images/Pipeline1.png "Pipeline"
[image6]: ./writeup_images/BirdsEyeView.png "Birds Eye View Tranform"
[image7]: ./writeup_images/Pipeline.png "Polinomial fit"
[image8]: ./writeup_images/deployBack.png "Deploy back into original image"
[image9]: ./examples/color_fit_lines.jpg "Fit Visual"
[image10]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup  

In this writeup I go through the mayer steps towards the completition of all the rubric points specified at the top.
You can follow the p4.ipynb jupyter notebook for implementation details.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is specified in the first cell of the Jupyter Notebook. In cell 2 we can find an example of raw image undistort image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I tried different methods and combinations of thresholds. I used a combination of color and gradient thresholds to generate a binary image. Following you can find a set of comparision images and corresponds to cells 3, 4 and 5

![alt text][image3]
![alt text][image4]
![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transform happends in the **birds_eye_view** method implemented in cell 6
The code for my perspective transform includes an 8 points interpolation (**cv2.INTER_LANCZOS4**) to minimize the loose of resolution. This improve significantly the pipeline and line detection. The source and destination points were hardcoded and hand selected using Mac Image editor for one of the straight lines sample images

```python
# Requires BGR Image
def birds_eye_view(img):
    # Pass in your image into this function
    # 1) Undistort using mtx and dist
    undist = getUndistortImage(img)
    # source points and dst points found via image editor
    src = np.float32([[220,660],[530,480],[940,660],[705,480]])
    dst = np.float32([[220,650],[220,60],[940,650],[940,60]])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    # Convert to grayscaly since warpPerspective needs it
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # using 8 points interpolation to minimize the loose of resolution (faster option: INTER_LINEAR)
    warped = cv2.warpPerspective(undist, M, gray.shape[::-1], flags=cv2.INTER_LANCZOS4)
    return warped, M, Minv
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 220, 660      | 220, 650      | 
| 530,480       | 220,60        |
| 940,660       | 940,650       |
| 705,480       | 940,60        |

By selecting this points for the transform I can efficiently select a region of interest, removing a lot of noise from the image to process.

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I implemented a Line class to simplify the process of line detection and curvature calculation. In this class I used the method **findLines** to find lines and fit a 2nd order polynomial. To improve polynomial accuracy and stabilitation we don't compute it based on a single frame but on the last 10 frames. This way we also avoid the possibility of non line detection.

```python
        # fit polynomious based on last 10 frames points
        self.lefty_a.append(lefty)
        self.leftx_a.append(leftx)
        self.rightx_a.append(rightx)
        self.righty_a.append(righty)

        lefty_tmp = []
        leftx_tmp = []
        righty_tmp = []
        rightx_tmp = []
        for i in range (max(1, len(self.lefty_a) - 10), len(self.lefty_a)):
            lefty_tmp.extend(self.lefty_a[i])
            leftx_tmp.extend(self.leftx_a[i])
            righty_tmp.extend(self.righty_a[i])
            rightx_tmp.extend(self.rightx_a[i])
```

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated within the Line class right in the findLines method already converted into real world meters. We store that radius in the Line and implement a method **getRealWorldCurve()** to return such valus. To calculate such transform to meters:

```python
        # Calculate curvature
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.left_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
 ```

And here the method:

```python
    # Curveture in Real World (meters)
    def getRealWorldCurve(self):
        return self.left_curvature, self.right_curvature
```
I do the same for the off the center calculation. Keeping stored the pixels off center and the number of pixels horizontally in the lane. Knowing that a lane is 3.7m we can do the correlation:

```python
    def get_off_center_meters(self):
        # lanes are 3.7m
        xm_per_pix = 3.7/self.pixels_in_lane
        return xm_per_pix * self.pixels_off_center
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented the method **deployBack()** to plot back all the calculations in the original image.

```python
# Warp the detected lane boundaries back onto the original image
def deployBack(src_img, warp_img, Minv, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warp_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (src_img.shape[1], src_img.shape[0])) 
    # Combine the result with the original image
    return cv2.addWeighted(getUndistortImage(src_img), 1, newwarp, 0.3, 0)
```

![alt text][image8]

An extra method is implemented to simplify the video processing. In such method I add the curvature and off center information to the frame. Also I can select different pipeline filters too for the image processing.

```python
def process_image(img):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # 1) Birds Eye View
    birds_img, M, Minv = birds_eye_view(img)
    # 2) Treat image
    #pipeline_image = pipeline(img, s_thresh=(90, 250), sx_thresh=(50, 100))
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # appliying perspective matrix
    #warp_img = cv2.warpPerspective(getUndistortImage(pipeline_image), perspective_M, gray.shape[::-1], flags=cv2.INTER_LANCZOS4)
    warp_img = getHLS(birds_img, 'SB', (90,250))
    # 2) Find Lines
    out_img, left_fit, right_fit, ploty, left_fitx, right_fitx = line.findLines(warp_img)
    # 3) Flip Back to original Image
    result = deployBack(img, warp_img, Minv, left_fitx, right_fitx, ploty)
    # Put curve info on the imager
    realWorldCurveLeft, realWorldCurveRight = line.getRealWorldCurve()
    text = "Radius of Curvature = {:.2f}(m)".format(realWorldCurveLeft)
    cv2.putText(result, text, (10,50), cv2.FONT_HERSHEY_PLAIN, fontScale=3, thickness=2, color=(255, 255, 255))
    off = line.get_off_center_meters()
    if off < 0:
        text = "Vehicule is {:.2f}(m) left of the center".format(abs(off))
    else:
        text = "Vehicule is {:.2f}(m) right of the center".format(abs(off))
    cv2.putText(result, text, (10,90), cv2.FONT_HERSHEY_PLAIN, fontScale=3, thickness=2, color=(255, 255, 255))
    return result
```

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_processed.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The approach I followed was to create an initial pipeline where I could complete every single step with one of the example images. I did try a lot of different combinations and images to calculate the best results for the treating the frame. I found out that the combination of **channel S in binary**, a **8 points interpolation** during the birds eye view and the **usage of 10 frames** to find lanes made the greates improvements. I wanted to include the curvature calculations for lines validation but I am running out of time. I would stimate an curvature error and discard frames where the detected curvature is out of range. At the same time, I believe that an extra color filter could help in harder frames. This is specially important during very winding roads since the 10 frames calculation wouldn't be that useful. We would probably have to reduce those 10 frames to 5 or less.
