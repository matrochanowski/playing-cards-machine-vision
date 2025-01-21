import cv2
import numpy as np
from object_detection import smart

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Błąd: Nie można otworzyć kamery.")
    exit()

while True:
    # Przechwyć klatkę
    ret, frame = cap.read()
    if not ret:
        print("Błąd: Nie można przechwycić klatki.")
        break

    # Wyświetl klatkę
    cv2.imshow("Kamera iPhone", frame)

    # Sprawdź, czy użytkownik nacisnął klawisz (np. Enter)
    key = cv2.waitKey(1)
    if key == 13:  # 13 to kod klawisza Enter
        height = frame.shape[0]  # Wysokość obrazu
        new_height = int(height * 0.8)  # 80% oryginalnej wysokości

        # Wytnij górne 80% obrazu
        frame_cropped = frame[:new_height, :]
        # Przeanalizuj klatkę
        data = smart(frame_cropped, visualise=True)

        # for entry in data:
        #     try:
        #         coordinates, card_info = entry
        #     except ValueError:
        #         continue
        #     x1, y1, x2, y2 = coordinates
        #     card_class, confidence = card_info
        #
        #     # Rysowanie prostokąta
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #
        #     # Pozycja tekstu
        #     text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)
        #
        #     # Dodanie tekstu
        #     cv2.putText(frame, card_class, text_position,
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        #
        # # Wyświetl klatkę z wynikami analizy
        # cv2.imshow("Analiza klatki", frame)

    # Zakończ program, jeśli użytkownik nacisnął klawisz 'q'
    if key == ord('q'):
        break

# Zwolnij zasoby
cap.release()
cv2.destroyAllWindows()

