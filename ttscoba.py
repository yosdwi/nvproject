from gtts import gTTS
import os

# Dictionary untuk melacak jumlah kemunculan setiap kata
word_counts = {}
# Variabel untuk menyimpan kata sebelumnya
previous_word = None


def texttospeech(display_text):
    global previous_word

    if display_text:
        input_display_text = display_text[-1]

        # Tambahkan kondisi sesuai kebutuhan Anda
        if input_display_text:
            # Inisialisasi hitungan kata jika kata belum ada dalam dictionary
            if input_display_text not in word_counts:
                word_counts[input_display_text] = 0
            
            # Tambahkan 1 ke hitungan kata
            word_counts[input_display_text] += 1

            # Jika kata sudah muncul 4 kali, mainkan suara
            #if word_counts[input_display_text] == 15 and input_display_text != previous_word:
            if word_counts[input_display_text] == 25 :
                # Inisialisasi objek gTTS
                tts = gTTS(text=input_display_text, lang='id')
                
                # Simpan file suara
                tts.save('output.mp3')

                # Mainkan file suara
                os.system('start output.mp3')

                # Reset hitungan kata
                word_counts[input_display_text] = 0
                previous_word = input_display_text
