
from gtts import gTTS
import speech_recognition as sr
import os


engine = sr.Recognizer()
mic = sr.Microphone()
hasil = ""
engine.pause_threshold = 0.5
def speechtotext():
    print("Merekam...")
    with mic as source:
        rekaman = engine.listen(source)

        try:
            hasil = engine.recognize_google(rekaman, language='id-ID')
            #print("Selesai...")
            print(hasil)
        except sr.UnknownValueError:
            print("Maaf suara Anda tidak terdeteksi sistem. Silahkan coba lagi.")
        except Exception as e:
            print(e)