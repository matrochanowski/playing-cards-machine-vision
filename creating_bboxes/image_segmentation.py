import numpy as np
import cv2
import os
import json
import random


def delete_all_files(folder_path: str):
    """
    Deletes all files in the specified folder.

    :param folder_path: Path to the folder whose files need to be deleted.
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder '{folder_path}' does not exist.")

    if not os.path.isdir(folder_path):
        raise ValueError(f"The path '{folder_path}' is not a directory.")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):  # Check if it's a file
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):  # Handle subdirectories if needed
                print(f"Skipping directory: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


def calculate_overlap_percentage(rect1, rect2):
    """
    Calculate the percentage of the area of the second rectangle that is inside the first rectangle.
    Handles coordinates in standard Cartesian space (y increases upwards).

    :param rect1: Coordinates of the first rectangle as (x1_top_left, y1_top_left, x1_bottom_right, y1_bottom_right)
    :param rect2: Coordinates of the second rectangle as (x2_top_left, y2_top_left, x2_bottom_right, y2_bottom_right)
    :return: Percentage of the area of the second rectangle inside the first rectangle.
    """
    # Normalize coordinates to handle Cartesian coordinate system
    x1_tl, y1_tl, x1_br, y1_br = rect1
    x2_tl, y2_tl, x2_br, y2_br = rect2

    # Ensure the top-left and bottom-right coordinates are properly ordered
    x1_tl, x1_br = min(x1_tl, x1_br), max(x1_tl, x1_br)
    y1_tl, y1_br = max(y1_tl, y1_br), min(y1_tl, y1_br)
    x2_tl, x2_br = min(x2_tl, x2_br), max(x2_tl, x2_br)
    y2_tl, y2_br = max(y2_tl, y2_br), min(y2_tl, y2_br)

    # Calculate intersection rectangle coordinates
    intersect_x_tl = max(x1_tl, x2_tl)
    intersect_y_tl = min(y1_tl, y2_tl)  # Top of intersection is the min of the top y's
    intersect_x_br = min(x1_br, x2_br)
    intersect_y_br = max(y1_br, y2_br)  # Bottom of intersection is the max of the bottom y's

    # Calculate intersection dimensions
    intersect_width = max(0, intersect_x_br - intersect_x_tl)
    intersect_height = max(0, intersect_y_tl - intersect_y_br)  # Notice the inverted y-axis logic here

    # Calculate the area of the intersection rectangle
    intersect_area = intersect_width * intersect_height

    # Calculate the area of the second rectangle
    rect2_width = x2_br - x2_tl
    rect2_height = y2_tl - y2_br  # Again, notice the Cartesian inversion
    rect2_area = rect2_width * rect2_height

    # Handle case when the second rectangle has zero area
    if rect2_area == 0:
        return 0

    # Calculate the percentage of the area of the second rectangle inside the first
    overlap_percentage = (intersect_area / rect2_area) * 100

    return overlap_percentage


def split_image(image: np.ndarray, num_parts: int):
    """
    Splits an image into an even number of parts and returns each part along with its coordinates.

    :param image: The input image as a NumPy array.
    :param num_parts: The number of parts to split the image into (must be even).
    :return: A list of tuples, each containing a NumPy array (image part) and its coordinates
             [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
    """
    # Get the image dimensions
    height, width, *channels = image.shape

    # Determine the number of rows and columns to split into
    sqrt_parts = int(np.sqrt(num_parts))
    if sqrt_parts ** 2 != num_parts:
        raise ValueError("The number of parts must allow a square grid division (e.g., 4, 16, 64).")

    rows = sqrt_parts
    cols = sqrt_parts

    # Check divisibility
    if height % rows != 0 or width % cols != 0:
        raise ValueError("The image dimensions must be divisible by the grid size.")

    # Calculate the size of each part
    part_height = height // rows
    part_width = width // cols

    # Split the image into parts and collect their coordinates
    parts_with_coords = []
    for i in range(rows):
        for j in range(cols):
            top_left_x = j * part_width
            top_left_y = i * part_height
            bottom_right_x = (j + 1) * part_width
            bottom_right_y = (i + 1) * part_height

            # Extract the part
            part = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Append the part and its coordinates
            parts_with_coords.append((part, [top_left_x, top_left_y, bottom_right_x, bottom_right_y]))

    return parts_with_coords


def split_image_uneven(image: np.ndarray, num_parts: int):
    """
    Splits an image into a specified number of parts with varying sizes, ensuring parts are more evenly distributed.

    :param image: The input image as a NumPy array.
    :param num_parts: The number of parts to split the image into.
    :return: A list of tuples, each containing a NumPy array (image part) and its coordinates
             [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
    """
    # Get the image dimensions
    height, width, *channels = image.shape

    # Validate the number of parts
    if num_parts < 1:
        raise ValueError("The number of parts must be at least 1.")

    # Determine approximate number of rows and columns
    rows = int(np.sqrt(num_parts))
    cols = (num_parts + rows - 1) // rows  # Ensure all parts fit

    # Generate row and column splits with slight variability
    row_splits = np.linspace(0, height, rows + 1, dtype=int)
    col_splits = np.linspace(0, width, cols + 1, dtype=int)

    # Introduce small random offsets to create variability, while ensuring valid splits
    max_offset = min(height // (2 * rows), width // (2 * cols))
    row_offsets = np.random.randint(-max_offset, max_offset + 1, size=len(row_splits) - 2)
    col_offsets = np.random.randint(-max_offset, max_offset + 1, size=len(col_splits) - 2)

    row_splits[1:-1] += row_offsets
    col_splits[1:-1] += col_offsets

    # Ensure splits are sorted and within bounds
    row_splits = np.clip(row_splits, 0, height)
    col_splits = np.clip(col_splits, 0, width)

    # Split the image and collect parts
    parts_with_coords = []
    for i in range(len(row_splits) - 1):
        for j in range(len(col_splits) - 1):
            top_left_x = col_splits[j]
            top_left_y = row_splits[i]
            bottom_right_x = col_splits[j + 1]
            bottom_right_y = row_splits[i + 1]

            # Extract the part
            part = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Append the part and its coordinates
            parts_with_coords.append((part, [top_left_x, top_left_y, bottom_right_x, bottom_right_y]))

    return parts_with_coords


def find_with_cuts(image_array, bboxes):
    parts = split_image(image_array, 64)
    for part, cords in parts:
        includes_object = False
        for bbox in bboxes:
            x1, y1, width, height = bbox
            x2, y2 = x1 + width, y1 + height
            percent_of_object = calculate_overlap_percentage(cords, [x1, y1, x2, y2])
            if percent_of_object > 25:
                includes_object = True
        if includes_object:
            cv2.imshow('Class', part)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('Nie zawiera żadnej klasy')


def make_cut_database(number=100, output_directory='cut_images', dataset='train'):
    json_filepath = os.path.join(os.pardir, 'jsons', 'images.json')
    with open(json_filepath, 'r') as file:
        data = json.load(file)
    for i, dict_entry in enumerate(data.values()):
        image_relative_path = dict_entry['path']
        image_path = os.path.join(os.pardir, dataset, 'images', image_relative_path)
        image_array = cv2.imread(image_path)

        annotations = dict_entry['annotations']
        annotations_list = list()
        for annotation in annotations:
            x1, y1, x2, y2 = annotation['bbox']
            annotations_list.append([x1, y1, x2, y2])

        def into_parts(how_many, trust=50):
            parts = split_image_uneven(image_array, how_many)
            j = 0
            for part, cords in parts:
                includes_object = False
                for bbox in annotations_list:
                    x1, y1, width, height = bbox
                    x2, y2 = x1 + width, y1 + height
                    percent_of_object = calculate_overlap_percentage(cords, [x1, y1, x2, y2])
                    if percent_of_object > trust:
                        includes_object = True

                if includes_object:
                    segment_path = f'{output_directory}/{1}_cut__{how_many}_{dataset}_{random.randint(1000, 9999)}.jpg'
                else:
                    segment_path = f'{output_directory}/{0}_cut_{how_many}_{dataset}_{random.randint(1000, 9999)}.jpg'

                try:
                    part = cv2.resize(part, (128, 128))
                except cv2.error:
                    break

                if includes_object:
                    cv2.imwrite(segment_path, part)
                else:
                    chance = random.randint(0, 40)
                    if chance == 3:
                        cv2.imwrite(segment_path, part)

                j += 1

        # CUT PICTURE IN 4 PARTS
        into_parts(4, trust=24)
        # # CUT PICTURE IN 16 PARTS
        # into_parts(16, trust=49)
        # # CUT PICTURE IN 64 PARTS
        # into_parts(64, trust=74)
        # CUT PICTURE IN RANDOM NR OF BIG PARTS
        for _ in range(1):
            into_parts(random.randint(3, 36), trust=90)
        # CUT PICTURE IN RANDOM NR OF SMALL PARTS
        for _ in range(1):
            into_parts(random.randint(37, 110), trust=90)
        # CUT PICTURE IN RANDOM NR OF SMALL PARTS
        for _ in range(3):
            into_parts(random.randint(110, 200), trust=90)
        if i >= number:
            break


def main():
    delete_all_files('cut_images')
    make_cut_database(dataset='train', output_directory='cut_images', number=1200)
    # print(calculate_overlap_percentage((2, 2, 4, -4), (1, 1, 3, 0)))

    # image = cv2.imread("first_image.jpg")
    # there_are_classes = [[447, 162, 61.343, 59.71], [400, 118, 41.372, 64.477], [318, 84, 45.407, 70.524],
    #                      [75, 248, 54.425, 73.628]]
    # image2 = cv2.imread('second_image.jpg')
    # find_with_cuts(image2, there_are_classes)
    # splitted = split_image(image, 4)
    #
    # for part, cords in splitted:
    #     print(f"{cords}")
    #     cv2.imshow("Image", part)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
