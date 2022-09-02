import cv2
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

path= "../citra-nir/100 meter/0004_100_n.JPG"
img_filtered = cv2.imread(path)
#img_filtered = img_1[1320:2552, 1640:3504]


# Detect faces
faces = face_cascade.detectMultiScale(img_filtered, 1.1, 4)
for (x, y, w, h) in faces:
  cv2.rectangle(img_filtered, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.namedWindow('Gambar Filtering_', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Gambar Filtering_", img_filtered)

cv2.namedWindow('face', cv2.WINDOW_KEEPRATIO)
cv2.imshow("face", img_filtered)
cv2.waitKey(0)