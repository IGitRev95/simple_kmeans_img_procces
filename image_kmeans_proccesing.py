import math
import time
import sys
import matplotlib.image as img
import numpy as np


# image = img.imread('MandrillMale-638x1024.jpg')
# image = img.imread('mandaril_light-weight.jpg')
# image = img.imread('Low-Poly-Colorful-Background-Preview.jpg')


def main(k,image_file_name):
    image = img.imread(image_file_name)
    image_pixel_array = np.array([pixel for row in image for pixel in row])

    print("step1 image to array of pixels - done")
    pixels_with_label_list = [[image_pixel_array[i], image_pixel_array[i], i] for i in
                              range(len(image_pixel_array))]  # ["label" , pixel (in RGB), pixel order]
    # keep the index for reserving the original order of the pixels
    print("step2 creating [label , pixel, order_idx] array - done")
    initTime = time.time()
    final_centers_with_assigned_pixels = random_init_kmeans(k, pixels_with_label_list)
    print("step3 create k-means with pixel pool - done")
    pixels_with_label_list_new_labels = assign_new_color(final_centers_with_assigned_pixels)
    print("step4 assign new labels / colors - done")

    re_colored_pixels_n_ordered = make_sorted_recolored_pixels_array(pixels_with_label_list_new_labels)

    print("step5 extraction of new pixels list - done")
    new_image = np.array(re_colored_pixels_n_ordered).reshape(image.shape)
    print("step6 reshape for construction of new image - done")
    img.imsave(f"re_colored_fork={k}.jpg", new_image)
    print("step7 new image file created - done")
    finishTime = time.time()
    print(f"finished in {finishTime-initTime} seconds")


def make_sorted_recolored_pixels_array(pixels_with_label_list_new_labels):
    pools = [centerNpool[1] for centerNpool in pixels_with_label_list_new_labels]
    pool_unite = []
    for pool in pools:
        for pixel in pool:
            pool_unite.append(pixel)
    re_colored_pixels = [[pix[0][2], pix[0][1]] for pix in pool_unite]
    re_colored_pixels.sort()
    re_colored_pixels_n_ordered = [pix[1] for pix in re_colored_pixels]
    return re_colored_pixels_n_ordered


def random_init_kmeans(k, pixels_with_label_list):
    # initialize k random clusters, assigned with empty sets
    centers_array = [[x, []] for x in np.random.randint(0, 256, size=(k, 3))]  # array of [center,bounded images]
    iteration_counter = 0
    cur_centers = [center[0] for center in centers_array]  # will keep current centers for comparison

    while True:
        iteration_counter = iteration_counter + 1
        print("iteration number {}\n".format(iteration_counter))

        #  Assigning step
        for pixel in pixels_with_label_list:
            centers_array[find_closest_center_index(centers_array, pixel[1])][1].append([pixel])
            # assign image to closest center

        #  Centers updating step
        for cluster in filter(lambda claster: len(claster[1]) > 0, centers_array):
            pixel_pool = [pixel[0][1] for pixel in cluster[1]]  # extracting all assigned pixels values
            cluster[0] = np.average(pixel_pool, axis=0)  # compute the average vector to be the new center
        new_centers = [center[0] for center in centers_array]
        #  check if the centers have changed in the past iteration - termination condition
        if np.array_equal(cur_centers, new_centers):
            break

        cur_centers = new_centers  # keeping current centers for comparison

        #  Images to clusters assigning reset
        for cluster in centers_array:
            cluster[1] = []
    # Normalizing centers values
    for center_centerPool in centers_array:
        center_centerPool[0] = [math.floor(val) for val in center_centerPool[0]]
        center_centerPool[0] = np.array(center_centerPool[0], dtype=np.uint8)

    # return the last centers with their final assigned pixels, centers normalized to floor and uint8
    return centers_array


def find_closest_center_index(center_array, data_unit):
    centers = map(lambda center_set_tuple: center_set_tuple[0], center_array)  # extract current centers
    squared_norm = lambda x: np.inner(x, x)
    x = [squared_norm(data_unit - center) for center in centers]  # compute norm distance from each center
    return np.argmin(x)  # return the index of the closest center


def assign_new_color(center_pixel_list):
    # for each pool, color/label any pixel with its assigned center
    for center_pixel_pool in center_pixel_list:
        cur_center = center_pixel_pool[0]
        cur_pool = center_pixel_pool[1]
        for pix in cur_pool:
            pix[0][1] = cur_center
    return center_pixel_list


if __name__ == "__main__":
    k = int(sys.argv[1])
    image_file_name = sys.argv[2]
    main(k,image_file_name)
