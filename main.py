import inline
import numpy as np
import matplotlib.pyplot as plt
import cv2  # This is the OpenCV Python library
import pytesseract  # This is the TesseractOCR Python library

carplate_img = cv2.imread('car.jpeg')
carplate_img_rgb = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
plt.imshow(carplate_img_rgb)

carplate_haar_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

def carplate_detect(image):
    carplate_overlay = image.copy()
    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay, scaleFactor=1.1, minNeighbors=3)
    for x, y, w, h in carplate_rects:
        cv2.rectangle(carplate_overlay, (x, y), (x + w, y + h), (255, 0, 0), 5)
    return carplate_overlay


detected_carplate_img = carplate_detect(carplate_img_rgb)
plt.imshow(detected_carplate_img)


def carplate_extract(image):
    global carplate_img
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in carplate_rects:
        carplate_img = image[y + 20:y + h - 5,
                       x + 4:x + w - 20]  # Adjusted to extract specific region of interest i.e. car license plate
    return carplate_img


def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


carplate_extract_img = carplate_extract(carplate_img_rgb)
carplate_extract_img = enlarge_img(carplate_extract_img, 150)
plt.imshow(carplate_extract_img)

carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)
plt.axis('off')
plt.imshow(carplate_extract_img_gray, cmap='gray')

carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray, 3)  # kernel size 3
plt.axis('off')
cv2.imshow('displaymywindows', carplate_extract_img_gray_blur)

print(pytesseract.image_to_string(carplate_extract_img_gray_blur,
                                  config=f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

cv2.waitKey(0)  # wait for a keyboard input
cv2.destroyAllWindows()