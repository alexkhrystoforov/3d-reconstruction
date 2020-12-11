import cv2

file_name = 'alpha-channel.png'
image = cv2.imread(file_name, -1)

print(image.shape)
IMG_RED, IMG_GREEN, IMG_BLUE, IMG_ALPHA = cv2.split(image)

cv2.namedWindow('r', cv2.WINDOW_NORMAL)
cv2.namedWindow('b', cv2.WINDOW_NORMAL)
cv2.namedWindow('g', cv2.WINDOW_NORMAL)
cv2.namedWindow('a', cv2.WINDOW_NORMAL)
cv2.imshow("r", IMG_RED)
cv2.imshow("g", IMG_GREEN)
cv2.imshow("b", IMG_BLUE)
cv2.imshow("a", IMG_ALPHA)
cv2.waitKey(0)