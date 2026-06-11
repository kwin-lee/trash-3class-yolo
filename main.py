from ultralytics import YOLO
import cv2
import os

#Loading Model

model = YOLO("best.pt")

class_names = {
    0: "organik",
    1: "anorganik",
    2: "b3"
}

#Deteksi Gambar

def detect_image(image_path):
    results = model(image_path, conf=0.25)

    annotated = results[0].plot()

    os.makedirs("hasil_prediksi", exist_ok=True)

    output_path = os.path.join(
        "hasil_prediksi",
        "hasil_" + os.path.basename(image_path)
    )

    cv2.imwrite(output_path, annotated)

    print("Hasil disimpan di:", output_path)

    cv2.imshow("Hasil Deteksi", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Deteksi Webcam

def detect_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam tidak terdeteksi")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame")
            break

        results = model(frame, conf=0.25)

        annotated = results[0].plot()

        cv2.imshow("Deteksi Sampah - Webcam", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

print("=== YOLO Trash Detection ===")
print("1. Deteksi gambar")
print("2. Deteksi webcam")

choice = input("Pilih menu (1/2): ")

if choice == "1":
    image_path = input("Masukkan path gambar: ")
    detect_image(image_path)

elif choice == "2":
    detect_webcam()

else:
    print("Pilihan tidak valid")