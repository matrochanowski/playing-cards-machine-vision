from test_recognition import classify_image
from image_segmentation import split_image, split_image_uneven
from cropped_image_recognition.test_classification import classify_card
from usage_image_json import id_to_class
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image
import time
import random
from collections import defaultdict
import os


def detect_card_classes(image_array, threshold=0.95):
    # split image in 4 parts
    parts = split_image(image_array, 4)
    for part, cords in parts:
        part_pil = Image.fromarray(part)
        # check for a card etiquete in the fragment
        answer, confidence = classify_image(part_pil)
        print(answer, confidence)
        if answer == 1:
            # split into more parts
            more_parts = split_image(part, 4)
            for another_part, another_cords in more_parts:
                part_pil = Image.fromarray(another_part)
                answer, confidence = classify_image(part_pil)
                print('smaller picture', answer, confidence)
                if answer == 1:
                    x_1, y_1, x_2, y_2 = another_cords
                    cv2.rectangle(image_array, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
                    which_card = classify_card(part)

                    text_position = (int(x_1), int(y_1) - 10)
                    cv2.putText(image_array, which_card, text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('image', image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def naive(image_array, threshold=0.85, split_into=74, visualise=True):
    start = time.time()
    parts = split_image_uneven(image_array, split_into)
    cards_bboxes_classes = []
    for part, cords in parts:
        part_pil = Image.fromarray(part)
        answer, place_confidence = classify_image(part_pil)
        if answer == 1 and place_confidence > threshold:
            x_1, y_1, x_2, y_2 = cords
            if visualise:
                cv2.rectangle(image_array, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
            which_card, confidence = classify_card(part)
            if confidence > 0.1:
                cards_bboxes_classes.append((cords, (which_card, confidence), place_confidence))

            if visualise:
                text_position = (int(x_1), int(y_1) - 10)
                cv2.putText(image_array, which_card, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    if visualise:
        end = time.time()
        print(end - start)
        cv2.imshow('image', image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cards_bboxes_classes


def smart(image_array, threshold=0.85, visualise=True, iterations=2):
    def quarter_approach(img, cords, tries=8):
        # check if there is an etiquette in the quarter
        img_pil = Image.fromarray(img)
        test_presence, confidence = classify_image(img_pil)
        if test_presence == 0:
            return []
        # check for a possibility that a quarter is etiquette
        test_if_etiquette, confidence = classify_card(img)
        if confidence == 1.0:
            return [cords]

        quarter_bboxes_classes = []
        # find etiquette's
        for i in range(tries):
            # use naive approach
            found_bboxes_classes = use_naive(img, cords, (18, 68))
            quarter_bboxes_classes += found_bboxes_classes
        return quarter_bboxes_classes

    def use_naive(img, current_fragment_cords, split_into):
        complete_arr = []
        n_splits = random.randint(split_into[0], split_into[1])
        found = naive(img, threshold=0.25, visualise=False, split_into=n_splits)
        for tup in found:
            if tup is None:
                return []
            coordinates = tup[0]
            current_x = current_fragment_cords[0]
            current_y = current_fragment_cords[1]
            coordinates[0] += current_x
            coordinates[2] += current_x
            coordinates[1] += current_y
            coordinates[3] += current_y
            complete_arr.append((coordinates, tup[1], tup[2]))
        return complete_arr

    def draw_boxes(image_array, data):
        for entry in data:
            try:
                coordinates, card_info, _ = entry
            except ValueError:
                continue
            x1, y1, x2, y2 = coordinates
            card_class, confidence = card_info

            # Rysowanie prostokąta
            cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Pozycja tekstu
            text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)

            # Dodanie tekstu
            cv2.putText(image_array, card_class, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Wyświetlanie obrazu
        cv2.imshow('Image with Cards', image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    start = time.time()
    to_draw = []
    for _ in range(iterations):
        # first divide into 4 parts for optimal
        _4_parts = split_image_uneven(image_array, 3)
        for quarter, quarter_cords in _4_parts:
            to_draw += quarter_approach(quarter, quarter_cords, tries=1)
        _4_parts = split_image_uneven(image_array, 4)
        for quarter, quarter_cords in _4_parts:
            to_draw += quarter_approach(quarter, quarter_cords, tries=2)
        _4_parts = split_image_uneven(image_array, 5)
        for quarter, quarter_cords in _4_parts:
            to_draw += quarter_approach(quarter, quarter_cords, tries=1)
    end = time.time()
    print(f'Czas {end - start}')
    probs_list, thre = calculate_card_probabilities(to_draw)
    to_return = [clas for clas, prob in probs_list if prob > thre]
    print("Karty znalezione na obrazku: ", to_return)
    if visualise:
        draw_boxes(image_array, to_draw)
    return to_return


def calculate_card_probabilities(cards):
    """
    Funkcja oblicza sumaryczne prawdopodobieństwa dla każdej klasy kart
    i zwraca je w posortowanej kolejności malejąco.

    :param cards: lista krotek, gdzie pierwszym elementem są koordynaty karty,
                  a drugim krotka z klasą karty i prawdopodobieństwem
    :return: lista krotek (klasa, sumaryczne prawdopodobieństwo) posortowana malejąco
    """
    # Słownik do przechowywania sum prawdopodobieństw dla każdej klasy
    class_probabilities = defaultdict(float)

    # Sumowanie prawdopodobieństw dla każdej klasy
    for _, (card_class, probability), probability_of_place in cards:
        if probability > 0.1:
            class_probabilities[card_class] += probability * probability_of_place

    # Sortowanie klas według łącznego prawdopodobieństwa malejąco
    sorted_classes = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
    mean = [prob for _, prob in sorted_classes if prob > sorted_classes[0][1] / 10]
    mean = sum(mean[1:]) / len(mean[1:])
    return sorted_classes, mean


if __name__ == '__main__':
    image_path1 = 'fourth_image.jpg'
    image_path2 = 'second_image.jpg'
    image_path3 = 'karty.png'
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    image3 = cv2.imread(image_path3)
    image5 = cv2.imread('karty.png')
    image6 = cv2.imread('sixth.jpg')
    image7 = cv2.imread('seventh.jpg')
    image8 = cv2.imread('eight.jpg')
    # detect_card_classes(image1)
    # print(smart(image1))
    print(smart(image7))
    smart(image2)
    smart(image3)
    smart(image6)
    smart(image8)
    # naive(image1)
    # naive(image2)
    # naive(image2)
    # naive(image3)
    # naive(image3)
    # naive(image5)
    # naive(image5)
    #
    # coordinates = split_image_uneven(image1, 74)
    #
    # # Draw rectangles on the original image to visualize the pieces
    # for part, coord in coordinates:
    #     start_height, start_width, end_height, end_width = coord
    #     # Draw a rectangle around each piece
    #     cv2.rectangle(image1, (start_width, start_height), (end_width, end_height), (0, 255, 0),
    #                   2)  # Green color, thickness=2
    #
    # # Display the image with rectangles
    # cv2.imshow("Image with Pieces", image1)
    # cv2.waitKey(0)  # Wait until a key is pressed
    # cv2.destroyAllWindows()  # Close all OpenCV windows
