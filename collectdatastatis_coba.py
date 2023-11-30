import os
import cv2
import keyboard
import time

cap = cv2.VideoCapture(1)
directory = 'Image_coba/'
capturing = False
current_letter = None
capture_start_time = time.time()  # Variable to store the capture start time
first_capture = True  # Flag to identify the first capture
delay_frames = 30  # Number of frames to wait before capturing (adjust as needed)

# Set the maximum capture duration (in seconds)
max_capture_duration = 50  # Change this value as needed

frame_count = 0

while True:
    _, frame = cap.read()

    count = {letter: len(os.listdir(directory + "/" + letter.upper())) for letter in 'abcdefghijklmnopqrstuvwxyz'}

    # Display the frame and the region of interest (ROI)
    cv2.imshow("data", frame)
    cv2.imshow("ROI", frame[200:500, 150:480])
    frame_roi = frame[200:500, 150:480]  # Capture the region of interest (ROI)

    interrupt = cv2.waitKey(10)

    # Tandai huruf yang akan di-"klik" secara otomatis
    if interrupt & 0xFF == ord(' '):
        if not capturing:
            frame_count = 0  # Reset the frame count
            capturing = True
            current_letter = 'b'
            print(f"Mulai menangkap gambar untuk huruf: {current_letter.upper()}")

    # Hentikan proses penangkapan ketika tombol spasi ditekan
    if keyboard.is_pressed('q'):
        if capturing:
            capturing = False
            print("Proses penangkapan dihentikan oleh pengguna")
            current_letter = None

    if capturing and current_letter is not None:
        if first_capture:
            time.sleep(2)
            first_capture = False
        else:
            frame_count += 1
            if frame_count >= delay_frames:
                # Simpan gambar sesuai dengan huruf yang sedang di-"klik"
                cv2.imwrite(directory + current_letter.upper() + '/' + str(count[current_letter.lower()]) + '.png', frame_roi)
                count[current_letter.lower()] += 1
                frame_count = 0  # Reset the frame count

                # Hentikan proses penangkapan setelah mencapai durasi maksimum
                elapsed_time = time.time() - capture_start_time
                if elapsed_time >= max_capture_duration:
                    capturing = False
                    print(f"Mencapai durasi maksimum ({max_capture_duration} detik). Proses penangkapan dihentikan.")
                    current_letter = None

cap.release()
cv2.destroyAllWindows()
