import json
import cv2
import os
import h5py
import random

current_dir = os.path.dirname(__file__)


def show_image_with_boxes(image_path, boxes, classes=None):
    """Displays an image with bounding boxes drawn around detected objects.

    Args:
        image_path (str): The path to the image file.
        boxes (list): A list of bounding boxes, each represented as a list [x, y, width, height].
        classes (list, optional): A list of class names corresponding to each bounding box. Defaults to None.
    """
    image = cv2.imread(image_path)

    if image is None:
        print("Nie udało się wczytać obrazu. Sprawdź ścieżkę.")
        return

    for i, box in enumerate(boxes):
        x, y, width, height = box

        top_left = (int(x), int(y))
        bottom_right = (int(x + width), int(y + height))

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        if classes is not None and i < len(classes):
            class_name = classes[i]
            text_position = (int(x), int(y) - 10)
            cv2.putText(image, class_name, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("Image with Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_cropped_images(image_path, boxes, output_folder, classes=None):
    """Crops regions of interest from an image based on bounding boxes and saves them as separate files.

    Args:
        image_path (str): Path to the image file.
        boxes (list): List of bounding boxes, each as [x, y, width, height].
        output_folder (str): Path to the directory where cropped images will be saved.
        classes (list, optional): List of class names for each box. If provided, filenames will include the class name. Defaults to None.
    """
    # Wczytaj obraz za pomocą OpenCV
    image = cv2.imread(image_path)

    # Sprawdź, czy obraz został poprawnie załadowany
    if image is None:
        print("Nie udało się wczytać obrazu. Sprawdź ścieżkę.")
        return

    # Upewnij się, że folder wyjściowy istnieje
    os.makedirs(output_folder, exist_ok=True)

    # Przejdź przez listę boxów i zapisz każdy wycięty fragment
    for i, box in enumerate(boxes):
        x, y, width, height = box
        x, y, width, height = int(x), int(y), int(width), int(height)

        # Wycięcie fragmentu obrazu według boxa
        cropped_image = image[y:y + height, x:x + width]

        # Zbuduj nazwę pliku dla wyciętego fragmentu
        if classes is not None and i < len(classes):
            class_name = classes[i]
            output_path = os.path.join(output_folder, f"{class_name}.jpg")
        else:
            output_path = os.path.join(output_folder, f"crop_{i}.jpg")

        # Zapisz wycięty fragment jako osobny plik
        cv2.imwrite(output_path, cropped_image)

    print(f"Zapisano wycięte obrazy w folderze: {output_folder}")


def get_uneven_cropped_images(image_path, boxes, classes):
    """Crops image regions defined by bounding boxes and returns them with their respective classes.

    Args:
        image_path (str): The path to the image file.
        boxes (list): A list of bounding boxes, each represented as [x, y, width, height].
        classes (list): A list of class labels corresponding to each bounding box.

    Returns:
        list: A list of tuples, where each tuple contains a cropped image (NumPy array) and its corresponding class label.
              Returns None if the image fails to load.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Nie udało się wczytać obrazu. Sprawdź ścieżkę.")
        return
    cropped_images = []
    for i, box in enumerate(boxes):
        x, y, width, height = box
        x, y, width, height = int(x), int(y), int(width), int(height)
        fifty_forty = random.randint(0, 2)
        if fifty_forty == 1:
            fifty_fifty2 = random.randint(0, 1)
            if fifty_fifty2 == 1:
                x += int(height * 0.02) * random.randint(0, 50)
                height += int(height * 0.02) * random.randint(0, 50)
            else:
                x -= int(height * 0.02) * random.randint(0, 25)
                height -= int(height * 0.02) * random.randint(0, 25)
        elif fifty_forty == 2:
            fifty_fifty2 = random.randint(0, 1)
            if fifty_fifty2 == 1:
                y += int(width * 0.02) * random.randint(0, 50)
                width += int(width * 0.02) * random.randint(0, 50)
            else:
                y -= int(width * 0.02) * random.randint(0, 25)
                width -= int(width * 0.02) * random.randint(0, 25)
        else:
            y += int(width * 0.02) * random.randint(0, 25)
            width += int(width * 0.02) * random.randint(0, 25)
            x += int(height * 0.02) * random.randint(0, 25)
            height += int(height * 0.02) * random.randint(0, 25)

        cropped_image = image[y:y + height, x:x + width]
        cropped_images.append((cropped_image, classes[i]))

    return cropped_images


def get_cropped_images(image_path, boxes, classes):
    """Crops image regions defined by bounding boxes and returns them with their respective classes.

    Args:
        image_path (str): The path to the image file.
        boxes (list): A list of bounding boxes, each represented as [x, y, width, height].
        classes (list): A list of class labels corresponding to each bounding box.

    Returns:
        list: A list of tuples, where each tuple contains a cropped image (NumPy array) and its corresponding class label.
              Returns None if the image fails to load.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Nie udało się wczytać obrazu. Sprawdź ścieżkę.")
        return
    cropped_images = []
    for i, box in enumerate(boxes):
        x, y, width, height = box
        x, y, width, height = int(x), int(y), int(width), int(height)
        cropped_image = image[y:y + height, x:x + width]

        cropped_images.append((cropped_image, classes[i]))

    return cropped_images


def get_cropped_images_from_matrix(image, boxes):
    """Crops regions from an image matrix based on provided bounding boxes.

    Args:
        image (numpy.ndarray): The image as a NumPy array.
        boxes (list): A list of bounding boxes, each as [x, y, width, height].

    Returns:
        list: A list of cropped image regions as NumPy arrays.
    """
    cropped_images = []
    for i, box in enumerate(boxes):
        x, y, width, height = box
        x, y, width, height = int(x), int(y), int(width), int(height)
        cropped_image = image[y:y + height, x:x + width]
        cropped_images.append(cropped_image)
    return cropped_images


def id_to_class(id):
    """Maps a class ID or a list of class IDs to their corresponding class names.

    Args:
        id (int or list):  A single class ID (int) or a list of class IDs.

    Returns:
        str or list: If the input 'id' is an integer, returns the corresponding class name (str).
                     If the input 'id' is a list of integers, returns a list of corresponding class names.
    """
    path = os.path.join(current_dir, 'jsons', 'categories.json')
    with open(path, 'r') as jsonfile:
        data = json.load(jsonfile)
    if isinstance(id, list):
        output = []
        for class_id in id:
            output.append(data[str(class_id)])
        return output
    if isinstance(id, int):
        return data[str(id)]


def load_all_cropped_images(dataset='train', less=21200):
    """Loads and crops all images and their bounding boxes from a dataset.

    Args:
        dataset (str, optional): The name of the dataset directory (e.g., 'train'). Defaults to 'train'.
        less (int, optional): The number of images to load. Defaults to 21200.

    Returns:
        list: A list of tuples, where each tuple contains a cropped image and its class label.
    """
    with open(os.path.join(current_dir, 'jsons', 'images.json'), 'r') as file:
        dict_data = json.load(file)
    images = []
    for i in range(0, less):
        file_path = os.path.join(current_dir, dataset, 'images', dict_data[str(i)]['path'])
        bboxes = []
        cats = []
        for content in dict_data[str(i)]['annotations']:
            bboxes.append(content['bbox'])
            cats.append(content['class_id'])
        to_extend = get_cropped_images(file_path, bboxes, cats)
        for img, clas in to_extend:
            img = cv2.resize(img, (64, 64))
            images.append((img, clas))

    return images


def yield_images_with_bboxes(file_path, less=21200, reshape_to=256):
    """Loads images and their bounding boxes from the specified dataset, resizing images to a given size.

    Args:
        file_path (str, optional): The dataset directory ('train' by default).
        less (int, optional): Number of images to load (21200 by default).
        reshape_to (int, optional): Size to resize images to (256x256 by default).

    Yields:
        tuple: A tuple containing a resized image and its corresponding bounding boxes.
    """
    with open(os.path.join(current_dir, 'jsons', 'images.json'), 'r') as file:
        dict_data = json.load(file)
    print('json file opened')

    for i in range(0, less):
        file_path = os.path.join(file_path, dict_data[str(i)]['path'])
        print(file_path)
        image = cv2.imread(file_path)
        image = cv2.resize(image, (reshape_to, reshape_to))

        card_classes = []
        for card_class in dict_data[str(i)]['annotations']:
            card_classes.append(id_to_class(card_class['class_id']))

        yield image, card_classes


def load_images_with_bboxes(dataset='train', less=21200, reshape_to=256):
    """Loads images and their bounding boxes from the specified dataset, resizing images to a given size.

    Args:
        dataset (str, optional): The dataset directory ('train' by default).
        less (int, optional): Number of images to load (21200 by default).
        reshape_to (int, optional): Size to resize images to (256x256 by default).

    Returns:
        tuple: A tuple containing two lists: the resized images and their corresponding bounding boxes.
    """
    with open(os.path.join(current_dir, 'jsons', 'images.json'), 'r') as file:
        dict_data = json.load(file)
    print('json file opened')
    images = []
    all_bboxes = []
    for i in range(0, less):
        file_path = os.path.join(current_dir, dataset, 'images', dict_data[str(i)]['path'])
        image = cv2.imread(file_path)
        image = cv2.resize(image, (reshape_to, reshape_to))
        images.append(image)
        bboxes = []
        for content in dict_data[str(i)]['annotations']:
            x, y, width, height = content['bbox']
            x = x * reshape_to / 640
            y = y * reshape_to / 640
            width = width * reshape_to / 640
            height = height * reshape_to / 640
            bboxes.append([x, y, width, height])
        all_bboxes.append(bboxes)

    return images, all_bboxes


def save_with_bboxes_to_hdf5(dataset='train', output_file='dataset.h5', less=21200, reshape_to=256):
    """Saves image data and bounding boxes to an HDF5 file.

    Args:
        dataset (str, optional): Source dataset directory ('train' by default).
        output_file (str, optional): Name of the output HDF5 file ('dataset.h5' by default).
        less (int, optional): Number of images to process (21200 by default).
        reshape_to (int, optional):  Image resizing dimension (256x256 by default).
    """
    with open(os.path.join(current_dir, 'jsons', 'images.json'), 'r') as file:
        dict_data = json.load(file)
    print('json file opened')
    path_database = os.path.join(current_dir, 'databases', output_file)
    with h5py.File(path_database, 'w') as h5file:
        images_group = h5file.create_group('images')
        bboxes_group = h5file.create_group('bboxes')

        for i in range(less):
            file_path = os.path.join(current_dir, dataset, 'images', dict_data[str(i)]['path'])
            image = cv2.imread(file_path)
            image = cv2.resize(image, (reshape_to, reshape_to))
            bboxes = []
            for content in dict_data[str(i)]['annotations']:
                x, y, width, height = content['bbox']
                x = x * reshape_to / 640
                y = y * reshape_to / 640
                width = width * reshape_to / 640
                height = height * reshape_to / 640
                bboxes.append([x, y, width, height])

            images_group.create_dataset(str(i), data=image, compression='gzip')
            bboxes_group.create_dataset(str(i), data=bboxes, compression='gzip')
            print(f'Saved {i} of {less}.')


def load_with_bboxes_from_hdf5(hdf5_file='dataset.h5'):
    """Loads image data and bounding boxes from an HDF5 file.

    Args:
        hdf5_file (str, optional): The name of the HDF5 file ('dataset.h5' by default).

    Yields:
        tuple:  A tuple containing the image and its corresponding bounding boxes.
    """
    path_database = os.path.join(current_dir, 'databases', hdf5_file)
    with h5py.File(path_database, 'r') as h5file:
        for i in range(len(h5file['images'])):
            image = h5file['images'][str(i)][()]
            bboxes = h5file['bboxes'][str(i)][()]
            yield image, bboxes


if __name__ == '__main__':
    images = get_uneven_cropped_images(
        'C:\\Users\\Mateusz\\PycharmProjects\\playing-cards-machine-vision\\train\\images', )
