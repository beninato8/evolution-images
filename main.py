import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np


class Gene:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def modify(self, img):
        if self.key == "rotate":
            return rotate_image(img, self.value)
        if self.key == "scale":
            return scale_image(img, self.value)
        return img


class Image:
    def __init__(self, data, genes):
        self.data = data
        self.genes = genes

    def getData(self):
        img = self.data
        for gene in self.genes:
            img = gene.modify(img)
        return img


def rotate_image(img, angle):
    size_reverse = np.array(img.shape[1::-1])  # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.0), angle, 1.0)
    MM = np.absolute(M[:, :2])
    size_new = MM @ size_reverse
    M[:, -1] += (size_new - size_reverse) / 2.0
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))


def scale_image(img, factor):
    height, width = img.shape[:2]
    return cv2.resize(img, (int(width * factor), int(height * factor)))


def main():
    original_image = cv2.imread("./images/tahoe.png", cv2.IMREAD_UNCHANGED)

    square_image = cv2.imread("./images/square.png", cv2.IMREAD_UNCHANGED)

    # get rid of alpha channel
    if square_image.shape[2] == 3:
        square_bgr = square_image
    else:
        square_bgr = square_image[:, :, :3]

    genes = [Gene("rotate", 45), Gene("scale", 5)]
    img = Image(square_bgr, genes)

    height, width = original_image.shape[:2]

    # apply all genes
    rotated_square = img.getData()

    overlay_x = 200
    overlay_y = -100

    # shift square image if negative coords
    if overlay_x < 0:
        rotated_square = rotated_square[:, abs(overlay_x) :]
        overlay_x = 0
    if overlay_y < 0:
        rotated_square = rotated_square[abs(overlay_y) :, :]
        overlay_y = 0

    rotated_height, rotated_width = rotated_square.shape[:2]

    if rotated_height == 0 or rotated_width == 0:
        # we are off the screen so no need to render
        return

    # deal with negatives
    max_overlay_x = width - rotated_width
    max_overlay_y = height - rotated_height

    overlay_x = min(overlay_x, max_overlay_x)
    overlay_y = min(overlay_y, max_overlay_y)

    # get region of interest chunk where overlay will replace
    roi = original_image[
        overlay_y : overlay_y + rotated_height,
        overlay_x : overlay_x + rotated_width,
    ]

    # merge the 2
    overlay = cv2.addWeighted(roi, 1, rotated_square, 1, 0)

    # put new overlay back
    original_image[
        overlay_y : overlay_y + rotated_height, overlay_x : overlay_x + rotated_width
    ] = overlay

    cv2.imshow("Overlayed Image", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
