from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = np.load("resources/test_data/0CE73E18-575A-4A70-9E40-F000B250344F.npz")
# cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
cv2.imshow("view", image)
cv2.waitKey(0)
# cv2.destroyWindow("view")