import numpy as  np
from matplotlib import pyplot as plt


def new_s(matrix, column, row):
    s_new = np.zeros((row, column))
    k = 20
    for i in range(column):
        if i < k:
            # print(Sr[i])
            s_new[i][i] = matrix[i]
        else:
            s_new[i][i] = 0

    return s_new



if __name__ == '__main__':
    img = plt.imread('noisy.jpg')
    print("noisy image")
    plt.imshow(img)
    plt.show()
    row, column, x = img.shape
    # RGB matrix
    red = np.zeros((row, column))
    green = np.zeros((row, column))
    blue = np.zeros((row, column))
    for j in range(row):
        for i in range(column):
            (red[j][i], green[j][i], blue[j][i]) = img[j][i]
    # SVD
    Ur, Sr, Vr = np.linalg.svd(red)
    Ub, Sb, Vb = np.linalg.svd(blue)
    Ug, Sg, Vg = np.linalg.svd(green)
    # make new S matrix
    Sr_new = new_s(Sr, column, row)
    Sg_new = new_s(Sg, column, row)
    Sb_new = new_s(Sb, column, row)


    # make clear matrix
    cleared_red = (Ur.dot(Sr_new)).dot(Vr)
    cleared_green = (Ug.dot(Sg_new)).dot(Vg)
    cleared_blue = (Ub.dot(Sb_new)).dot(Vb)

    # print new img
    new_img = np.zeros((row, column, 3))
    for i in range(row):
        for j in range(column):
            cleared_red[i][j] = cleared_red[i][j] / 256
            cleared_green[i][j] = cleared_green[i][j] / 256
            cleared_blue[i][j] = cleared_blue[i][j] / 256
            new_img[i][j] = (cleared_red[i][j], cleared_green[i][j], cleared_blue[i][j])

    Image = np.clip(new_img, 0, 1)
    print("cleared image")

    plt.imshow(Image)
    plt.show()
    print("saving ")
    plt.imsave('cleared.jpeg', Image)


