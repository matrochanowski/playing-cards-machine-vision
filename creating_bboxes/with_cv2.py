import cv2
import numpy as np
from matplotlib import pyplot as plt
from usage_image_json import save_cropped_images


img_rgb = cv2.imread('cards.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('A-template.png', cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_rgb, top_left, bottom_right, 255, 1)

plt.subplot(121), plt.imshow(res, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_rgb, cmap='gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

plt.show()

threshold = 0.6
loc = np.where(res >= threshold)

suma = 0
for i, pt in enumerate(zip(*loc[::-1])):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    suma += 1
print(suma)
cv2.imwrite('cards-found.jpg', img_rgb)
