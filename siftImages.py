import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt


# Count the arguments
arguments = len(sys.argv) - 1
def matrix_distance(a_list, b_list):
    sum_ = 0
    for x, y in zip(a_list, b_list):
        result = (x - y) ** 2
        sum_ += result
    return (sum_) ** (1 / 2)

def Task2(Images):
    K_value=[5, 10, 20]
    Total_key_points =[];
    for each_K in K_value:
        normalized_list = []
        dissimilarity_list = []
        sift = cv2.xfeatures2d.SIFT_create()
        for each_image in Images:
            descriptor_list = []
            image = cv2.imread(each_image, 0)
            # Aspect ratio
            width = 600
            height = 480
            dim = (width, height)
            # resize image
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            kp, des = sift.detectAndCompute(resized, None)
            # displaying keypoints
            if(each_K==5):
                print("# of keypoints in ", each_image, " is ", len(kp))
                Total_key_points.append(len(kp))
            descriptor_list.append(des)
            descriptors = np.array(descriptor_list[0])
            #stack the sequence of input arrays vertically to make a single array.
            for descriptor in descriptor_list[1:]:
                descriptors = np.vstack((descriptors, descriptor))
            #K-means clustering
            clusters=int(len(kp) * (each_K / 100))
            Z = np.float32(descriptors)#reshaping the image
            Image_features = np.array([np.zeros(clusters) for i in range(1)])
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)#defining criteria
            ret, label, center = cv2.kmeans(Z, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)# calling K-means function which returns 3 values
            for i in range(1):
                for idx in range(0, clusters):
                    Image_features[i][idx] = len(Z[label.ravel() == idx])
            Image_features_1 = [[i / sum(j) for i in j] for j in Image_features]
            normalized_list.append(Image_features_1)
            #plot Histogram
            x_scalar = np.arange(clusters)
            y_scalar = np.array([abs(np.sum(Image_features[:, h], dtype=np.int32)) for h in range(clusters)])
            plt.bar(x_scalar, y_scalar)
            plt.title("occurrence of the visual words")
            plt.xlabel("Visual Words")
            plt.ylabel("Frequency of Words")
            plt.xticks(x_scalar + 0.4, x_scalar)
            plt.show()
            #generating dissimilarity matrix based on normalized_list
            for i in normalized_list:
                foo = [matrix_distance(i[0], j[0]) for j in normalized_list]
                dissimilarity_list.append(foo)
        length=len(Images)
        if (each_K == 5):
            print()
            print("for k =",each_K,"%","(total number of keypoionts)=",sum(Total_key_points)*0.05)
        if (each_K == 10):
            print()
            print("for k =",each_K,"%","(total number of keypoionts)=",sum(Total_key_points)*0.10)
        if (each_K == 20):
            print()
            print("for k =",each_K,"%","(total number of keypoionts)=",sum(Total_key_points)*0.20)
        #displaying mastrix in user readable format
        print("   ", end="")
        for k in range(1, length + 1):
            print(str(k) + "     ", end="")
        print()
        if (k == 15):
            flag1 = 76
            flag2 = 75
        elif (k == 14):
            flag1 = 64
            flag2 = 63
        elif (k == 13):
            flag1 = 53
            flag2 = 52
        elif (k == 12):
            flag1 = 43
            flag2 = 42
        elif (k == 11):
            flag1 = 34
            flag2 = 33
        elif (k == 10):
            flag1 = 26
            flag2 = 25
        elif (k == 9):
            flag1 = 19
            flag2 = 18
        elif (k == 8):
            flag1 = 13
            flag2 = 12
        elif(k==7):
            flag1=8
            flag2=7
        elif(k==6):
            flag1=4
            flag2=3
        elif(k==5):
            flag1=1
            flag2=0
        elif (k == 4):
            flag1 = -1
            flag2=0
        elif (k == 3 or k==2):
            flag1 = -2
            flag2=0
        else:
            flag1 = -1000
            flag2 = 1000
        count = 1
        temp=1
        for i in dissimilarity_list:
            if (count >= arguments + k+flag1  and count <= arguments + k + arguments+flag2):
                print(str(temp) + "  ", end="")
                temp += 1
            for j in i:
                if (count>=arguments+k+flag1 and count<=arguments+k+arguments+flag2):
                    print(str(round(j, 2)) + "  ", end="")
            if (count >= arguments + k+flag1  and count <= arguments + k + arguments+flag2):
                print()
            count += 1


def Task1(argument1):
    image = cv2.imread(argument1)
    # Aspect ratio
    width = 600
    height = 480
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    resized_copy = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    #image spliting
    Y, Cr, Cb = cv2.split(gray)
    #Extracting Luminance Y component and resizing
    Y_component = cv2.resize(Y, dim, interpolation=cv2.INTER_AREA)
    # creating SIFT and keypoints detection for Luminance Y component
    sift = cv2.SIFT_create()
    keypoints_1 = sift.detect(Y_component, None)

    img_1 = cv2.drawKeypoints(Y_component, keypoints_1, resized_copy, (255, 0, 0), flags=4)
    for i in range(0, len(keypoints_1)):
        img_final = cv2.drawMarker(img_1, (int(keypoints_1[i].pt[0]), int(keypoints_1[i].pt[1])), color=(0, 255, 0),
                                   markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)
    # concatanate image Horizontally
    img_concate_Hori = np.concatenate((resized, img_final), axis=1)
    cv2.imwrite('Final.jpg', img_concate_Hori)
    cv2.namedWindow('Final.jpg', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Final.jpg', 1080, 720)
    cv2.imshow('Final.jpg', img_concate_Hori)
    print(len(keypoints_1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def parse_and_run():
    if (arguments == 1):
        Task1(sys.argv[1])
    else:
        Task2(sys.argv[1:])


if __name__ == '__main__':
    parse_and_run()