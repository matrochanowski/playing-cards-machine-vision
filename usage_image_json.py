import json
import cv2
import os


def show_image_with_boxes(image_path, boxes, classes=None):
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


def get_cropped_images(image_path, boxes, classes):
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


def id_to_class(id):
    with open('jsons/categories.json', 'r') as jsonfile:
        data = json.load(jsonfile)
    if isinstance(id, list):
        output = []
        for class_id in id:
            output.append(data[str(class_id)])
        return output
    if isinstance(id, int):
        return data[str(id)]


def load_all_images(dataset='train', less=21200):
    images = []
    with open('jsons/images.json', 'r') as file:
        dict_data = json.load(file)
    for i in range(0, less):
        file_path = f'{dataset}\\images\\{dict_data[str(i)]['path']}'
        bboxes = []
        cats = []
        for content in dict_data[str(i)]['annotations']:
            bboxes.append(content['bbox'])
            cats.append(content['class_id'])
        images.extend(get_cropped_images(file_path, bboxes, cats))

    return images


if __name__ == '__main__':
    with open('jsons/images.json', 'r') as jsonfile:
        data = json.load(jsonfile)
        # bierzemy obrazek o ID '10000'
        id = '10000'
        path = data[id]['path']
        boxes = []
        classes = []
        for element in data[id]['annotations']:
            boxes.append(element['bbox'])
            classes.append(element['class_id'])
        absolute_path = f'train\\images\\{path}'
        print(classes)
        print(id_to_class(classes))
        show_image_with_boxes(absolute_path, boxes, classes=id_to_class(classes))
        save_cropped_images(absolute_path, boxes, 'cropped\\images', classes=id_to_class(classes))
