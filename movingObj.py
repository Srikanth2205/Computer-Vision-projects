# Import libraries
import cv2
import numpy as np
import sys
import os

# main function which takes the input
def Task1(filename):
    cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(filename))
    # Reading the first frame for background subtraction
    ret_, image_frame = cap.read()
    # converting to float
    float_cvt = np.float32(image_frame)
    back_img = None

    # check whether the given file is opened
    if not cap.isOpened:
        print('error in file opening: ')
        exit(0)
    count = 0
    while True:
        # reading the frames one by one
        ret, frame = cap.read()
        # break condition if frame is none
        if frame is None:
            break
        # this function updates the running average
        cv2.accumulateWeighted(frame, float_cvt, 0.02)
        # Back ground subtracted image is created
        background_image = cv2.convertScaleAbs(float_cvt)

        backGround_img_conversion = background_image.copy()
        # Converting to BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Remove Noise using GaussianBlur
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if back_img is None:
            back_img = gray
            continue
        # Finding the difference to find the moving object
        difference_of_frames = cv2.absdiff(back_img, gray)

        # Binary Threshold to increase the contour

        Binary_threshold = cv2.threshold(difference_of_frames, 30, 255, cv2.THRESH_BINARY)[1]
        BT_copy = Binary_threshold.copy()

        # Creating kernel to remove noises and small objects from the frame
        kernel = np.ones((5, 5), np.uint8)
        Binary_threshold = cv2.morphologyEx(Binary_threshold, cv2.MORPH_OPEN, kernel)

        # Dilating the white area to get much area of objects

        Binary_threshold = cv2.dilate(Binary_threshold, None, iterations=4)

        # Finding the contours to create the estimated motion of the moving objects
        contours, _ = cv2.findContours(Binary_threshold.copy(),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize for classification of the moving objects
        others_cnt = 0
        cars_cnt = 0
        people_cnt = 0
        loop_count = 0

        # Iterating through all the contours

        for eachcontour in contours:
            # Eliminating small objects
            if cv2.contourArea(eachcontour) < 1000:
                continue
            (_, _, X, Y) = cv2.boundingRect(eachcontour)

            loop_count += 1

            # classifying moving objects
            area = float(X) / Y
            if 0.3 <= area <= 1.0:
                tpe = "person"
                people_cnt += 1
            elif area > 1.0:
                tpe = "car"
                cars_cnt += 1
            else:
                tpe = "other"
                others_cnt += 1
        count =display_task1(count, loop_count, people_cnt, cars_cnt, others_cnt,
                      Binary_threshold, backGround_img_conversion,
                      frame, BT_copy)
        cv2.waitKey(30)
    cv2.destroyAllWindows()
    # delete temporary files
    os.remove("pic1.jpg")
    os.remove("pic2.jpg")
    os.remove("pic3.jpg")
    os.remove("Final.jpg")


def Task2(filename):
    cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(filename))
    # Reading the first frame for background subtraction
    ret_, image_frame = cap.read()
    # converting to float
    float_cvt = np.float32(image_frame)
    back_img = None
    older_frame = None

    # check whether the given file is opened
    if not cap.isOpened:
        print('error in file opening: ')
        exit(0)
    count = 0
    while True:
        # reading the frames one by one
        ret, frame = cap.read()
        # break condition if frame is none
        if frame is None:
            break
        image_tracking = frame.copy()
        # this function updates the running average
        cv2.accumulateWeighted(frame, float_cvt, 0.02)
        # Converting to BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Remove Noise using GaussianBlur
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if back_img is None:
            older_frame = frame.copy()
            back_img = gray
            continue
        # Finding the difference to find the moving object
        difference_of_frames = cv2.absdiff(back_img, gray)

        # Binary Threshold to increase the contour

        Binary_threshold = cv2.threshold(difference_of_frames, 30, 255, cv2.THRESH_BINARY)[1]

        # Creating kernel to remove noises and small objects from the frame
        kernel = np.ones((5, 5), np.uint8)
        Binary_threshold = cv2.morphologyEx(Binary_threshold, cv2.MORPH_OPEN, kernel)

        # Dilating the white area to get much area of objects

        Binary_threshold = cv2.dilate(Binary_threshold, None, iterations=4)

        # Finding the contours to create the estimated motion of the moving objects
        contours, _ = cv2.findContours(Binary_threshold.copy(),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize for classification of the moving objects
        others_cnt = 0
        cars_cnt = 0
        people_cnt = 0
        loop_count = 0

        for eachcontour in contours:
            # Eliminating small objects
            if cv2.contourArea(eachcontour) < 1000:
                continue
            (A, B, C, D) = cv2.boundingRect(eachcontour)


            image_tracking = cv2.arrowedLine(image_tracking, (A - C, B - D), (A, B),
                                        (255, 0, 0), 2)
            loop_count += 1

            # classifying moving objects
            area = float(C) / D
            if 0.3 <= area <= 1.0:
                tpe = "person"
                people_cnt += 1
            elif area > 1.0:
                tpe = "car"
                cars_cnt += 1
            else:
                tpe = "other"
                others_cnt += 1
        older_frame,count=display_task2(count,loop_count,people_cnt,cars_cnt,others_cnt,Binary_threshold,older_frame,frame,image_tracking)

    cv2.destroyAllWindows()
    #delete temporary files
    os.remove("pic1.jpg")
    os.remove("pic2.jpg")
    os.remove("pic3.jpg")
    os.remove("Final.jpg")
    return 0

def display_task1(count,loop_count,people_cnt,cars_cnt,others_cnt,Binary_threshold,backGround_img_conversion,frame,BT_copy):
    print("Frame {}: {} objects ".format(str(count), str(loop_count)), end="")
    print("({} persons, {} car and {} others)".format(str(people_cnt), str(cars_cnt), str(others_cnt)))
    # Converting to binary
    binary_img = cv2.threshold(Binary_threshold, 0, 255, cv2.THRESH_BINARY)[1]
    count += 1
    # Applying the Connected components algorithm on the frame
    retval, labels = cv2.connectedComponents(binary_img)
    label_hue = np.uint8(179 * labels / np.max(labels))
    # writing to a temperary .jpg file to concatinate the frame in a single window
    cv2.imwrite('pic1.jpg', backGround_img_conversion)
    out1 = cv2.imread('pic1.jpg')
    img_concate_Hori2 = np.concatenate((frame, out1), axis=1)
    # writing to a temperary .jpg file to concatinate the frame in a single window
    cv2.imwrite('pic3.jpg', BT_copy.copy())
    out3 = cv2.imread('pic3.jpg')

    frame[label_hue == 0] = 0
    # writing to a temperary .jpg file to concatinate the frame in a single window
    cv2.imwrite('pic2.jpg', frame.copy())
    img_concate_Hori1 = np.concatenate((out3, frame), axis=1)
    # vertical concatination of two horizontal concatination to fit into a single window
    img_concate_Verti = np.concatenate((img_concate_Hori2, img_concate_Hori1), axis=0)

    # writing to temperary .jpg file with final vertical concatinated image and displaying in single window
    cv2.imwrite('Final.jpg', img_concate_Verti)
    cv2.namedWindow('Final.jpg', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Final.jpg', 1080, 720)
    cv2.imshow('Final.jpg', img_concate_Verti)
    return count

def display_task2(count,loop_count,people_cnt,cars_cnt,others_cnt,Binary_threshold,older_frame,frame,image_tracking):
    print("Frame {}: {} objects ".format(str(count), str(loop_count)), end="")
    print("({} persons, {} car and {} others)".format(str(people_cnt), str(cars_cnt), str(others_cnt)))
    # Converting to binary
    binary_img = cv2.threshold(Binary_threshold, 0, 255, cv2.THRESH_BINARY)[1]
    count += 1
    # Applying the Connected components algorithm on the frame
    retval, labels = cv2.connectedComponents(binary_img)
    label_hue = np.uint8(179 * labels / np.max(labels))
    # writing to a temperary .jpg file to concatinate the frame in a single window
    cv2.imwrite('pic1.jpg', older_frame)
    out1 = cv2.imread('pic1.jpg')
    #horizontal concatination with the original frame
    img_concate_Hori2 = np.concatenate((frame, out1), axis=1)

    # writing to a temperary .jpg file to concatinate the frame in a single window
    cv2.imwrite('pic3.jpg', image_tracking.copy())
    out3 = cv2.imread('pic3.jpg')

    older_frame = frame.copy()
    frame[label_hue == 0] = 0
    # writing to a temperary .jpg file to concatinate the frame in a single window
    cv2.imwrite('pic2.jpg', frame.copy())
    img_concate_Hori1 = np.concatenate((out3, frame), axis=1)
    # vertical concatination of two horizontal concatination to fit into a single window
    img_concate_Verti = np.concatenate((img_concate_Hori2, img_concate_Hori1), axis=0)

    # writing to temperary .jpg file with final vertical concatinated image and displaying in single window
    cv2.imwrite('Final.jpg', img_concate_Verti)
    cv2.namedWindow('Final.jpg', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Final.jpg', 1080, 720)
    cv2.imshow('Final.jpg', img_concate_Verti)
    cv2.waitKey(30)
    return older_frame, count


def parse_and_run():
    if (sys.argv[1] == '-b'):
        Task1(sys.argv[2])
    elif(sys.argv[1] == '-s'):
        Task2(sys.argv[2])
    else:
        print("invalid input!!!!\nagrument 1 should be either '-b' or '-s' ")
        exit(1)


if __name__ == '__main__':
    parse_and_run()