import cv2
import numpy as np
import sys

# Count the arguments
arguments = len(sys.argv) - 1
if(arguments<2or arguments>2):
    print("invalid  arguments")
    sys.exit(1)

position = 1
Task1_flag=0
while (arguments >= position):
    if(sys.argv[position]=="-XYZ"):
        print ("Choosen Colour space: %s" % (sys.argv[position]))
        position = position + 1
        Task1_flag=1
    elif (sys.argv[position] == "-Lab"):
        print("Choosen Colour space: %s" % (sys.argv[position]))
        position = position + 1
        Task1_flag = 1;
    elif (sys.argv[position] == "-YCrCb"):
        print("Choosen Colour space: %s" % (sys.argv[position]))
        position = position + 1
        Task1_flag = 1;
    elif (sys.argv[position] == "-HSV"):
        print("Choosen Colour space: %s" % (sys.argv[position]))
        position = position + 1
        Task1_flag = 1;
    else:
        break

def main(argument1, argument2):
    if (Task1_flag!=0):
        image=cv2.imread(argument2)
        Task1(image)
    else:
        image1 = cv2.imread(argument1)
        image2 = cv2.imread(argument2)
        white_background(image1, image2)

def Task1(image):
    """
    Function to display image
    :param image:
    :return:
    """

    if(sys.argv[1]=='-XYZ'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    elif(sys.argv[1]=='-Lab'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    elif(sys.argv[1]=='-YCrCb'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif(sys.argv[1]=='-HSV'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    B,G,R=cv2.split(gray)
    cv2.imwrite('pic1.jpg', B)
    cv2.imwrite('pic2.jpg', G)
    cv2.imwrite('pic3.jpg', R)
    out1 = cv2.imread('pic1.jpg')
    out2 = cv2.imread('pic2.jpg')
    out3 = cv2.imread('pic3.jpg')
    #concatanate image Horizontally
    img_concate_Hori1=np.concatenate((image,out3),axis=1)
    img_concate_Hori2=np.concatenate((out2,out1),axis=1)
    #concatanate image Vertically
    img_concate_Verti=np.concatenate((img_concate_Hori1,img_concate_Hori2),axis=0)

    cv2.imwrite('Final.jpg', img_concate_Verti)
    cv2.namedWindow('Final.jpg', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Final.jpg', 1080, 720)
    cv2.imshow('Final.jpg', img_concate_Verti)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def parse_and_run():
    argument1 = sys.argv[1]
    argument2 = sys.argv[2]
    main(argument1,argument2)

def white_background(image1,image2):
    # convert to hsv
    hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    height=image1.shape[0]
    width=image1.shape[1]
    # threshold using inRange
    range1 = (36, 95, 40)
    range2 = (86, 255, 255)
    mask = cv2.inRange(hsv, range1, range2)

    mask = cv2.bitwise_not(mask)

    # apply morphology opening to mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # load background (could be an image too)
    Back_ground = np.full(image1.shape, 255, dtype=np.uint8)  # white Back_ground

    # get masked foreground
    foreground_mask = cv2.bitwise_and(image1, image1, mask=mask)

    # get masked background, mask must be inverted
    mask = cv2.bitwise_not(mask)
    Back_ground_mask = cv2.bitwise_and(Back_ground, Back_ground, mask=mask)

    # combine masked foreground and masked background
    final = cv2.bitwise_or(foreground_mask, Back_ground_mask)
    mask = cv2.bitwise_not(mask)  # revert mask to original

    result = image1.copy()
    result[mask == 0] = (255, 255, 255)

    # write result to disk
    cv2.imwrite("Image.png", mask)
    cv2.imwrite("Image_green2white.jpg", final)

    cv2.imwrite("White_background.jpg", result)
    white_image = cv2.imread("White_background.jpg")
    img_concate_Hori1 = np.concatenate((image1, white_image), axis=1)
    cv2.imwrite('out1.jpg', img_concate_Hori1)
    graphical_background(image1,image2,height,width,img_concate_Hori1)


def graphical_background(image1,image2,height,width,img_concate_Hori1):
    print('This image1 is:', type(image1),
          ' with dimensions:', image1.shape)
    image_copy1 = np.copy(image1)
    # Defining colour threshold
    lower_green = np.array([30, 160, 0])
    upper_green = np.array([220, 255, 145])
    #creating mask
    mask = cv2.inRange(image_copy1, lower_green, upper_green)
    masked_image = np.copy(image_copy1)
    masked_image[mask != 0] = [0, 0, 0]
    dim = (width, height)
    # resize image
    resized = cv2.resize(image2, dim, interpolation=cv2.INTER_AREA)
    background_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    crop_background = background_image[0:height, 0:width]
    crop_background[mask == 0] = [0, 0, 0]
    complete_image = masked_image + crop_background
    cv2.imwrite('Image_with_background.jpg',complete_image)
    img_bg=cv2.imread('Image_with_background.jpg')
    img_concate_Hori2 = np.concatenate((resized, img_bg), axis=1)
    img_concate_Verti = np.concatenate((img_concate_Hori1, img_concate_Hori2), axis=0)
    cv2.imwrite('Final2.jpg', img_concate_Verti)
    cv2.namedWindow('Final2.jpg', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Final2.jpg', 1080, 720)
    cv2.imshow('Final2.jpg', img_concate_Verti)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    parse_and_run()