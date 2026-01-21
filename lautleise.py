import librosa
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------
# Ordner
# -----------------------------------------------------
input_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\Input"
output_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\output"

os.makedirs(output_folder, exist_ok=True)

# -----------------------------------------------------
# Erlaubte Dateiendungen
# -----------------------------------------------------
valid_ext = (".mp3", ".ogg", ".wav")

# -----------------------------------------------------
# Analyse-Funktion (hier kommt dein kompletter Code rein)
# -----------------------------------------------------
def analyse_audio_file(filepath):

    audio_file = os.path.basename(filepath)
    print("\nVerarbeite:", audio_file)

    # Audio laden
    y, sr = librosa.load(filepath, sr=None, mono=True)
    #Parameter
    frame_length = int(0.05 * sr)
    hop_length = int(0.001 * sr)

    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop_length))#Fouriertransformation

    def avg_spectrum(stft, center_index, window=10):#durchschnittliches Frequenzspektrum  um center_index
        half = window // 2
        start = max(center_index - half, 0)
        end   = min(center_index + half, stft.shape[1])
        return np.mean(stft[:, start:end], axis=1)

    max_amp = np.max(np.abs(y))
    high_thresh = 0.2 * max_amp

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    max_abs_per_frame = np.max(np.abs(frames), axis=0)

    peak = []
    drop = []
    i = 150
    ton = False

    while i < len(max_abs_per_frame) - 300:

        if max_abs_per_frame[i] < high_thresh:
            ton = False
        if max_abs_per_frame[i] > high_thresh and not ton:
            i += 250 #Überspringen von 250 indizes wenn das Rauschen aufhöht
            ton = True

        if max_abs_per_frame[i + 150] > high_thresh and ton:

            segment_next = max_abs_per_frame[i+100:i+300]
            segment_prev = max_abs_per_frame[i-250:i-50]
            #-----
            # Die Lautstärkeanderung bleibt immer ein moment stabil cond_next_Stable und prev stesten das.
            #-----
            cond_next_stable = np.all(
                (segment_next >= 0.75 * max_abs_per_frame[i+100]) &     
                (segment_next <= 1.5 * max_abs_per_frame[i+100])
            )
            cond_prev_stable = np.all(
                (segment_prev >= 0.75 * max_abs_per_frame[i-20]) &
                (segment_prev <= 1.5 * max_abs_per_frame[i-20])
            )

            # ---- Rise ----
            if (
                max_abs_per_frame[i+50] >= 1.33 * max_abs_per_frame[i]
                and cond_next_stable and cond_prev_stable
            ):
                S1 = avg_spectrum(stft, i)
                S2 = avg_spectrum(stft, i + 50)
                flux = np.sum(np.abs(S2 - S1)) #Vergleich der Spektren bei lauterstärkeänderung
                if flux < 50:                   #Frequenzänderung uss bei den gesuchten Lautstärkeänderungen gering sein.
                    peak.append(i+75)
                    i += 100
                    continue

            # ---- Drop ----
            if (
                max_abs_per_frame[i+50] <= 0.75 * max_abs_per_frame[i]
                and cond_next_stable and cond_prev_stable
            ):
                S1 = avg_spectrum(stft, i)
                S2 = avg_spectrum(stft, i + 50)
                flux = np.sum(np.abs(S2 - S1))#Vergleich der Spektren bei lauterstärkeänderung
                if flux < 50:                   #Frequenzänderung uss bei den gesuchten Lautstärkeänderungen gering sein.
                    drop.append(i+25)
                    i += 100
                    continue

        i += 1

    # Ergebnisse
    peak = np.array(peak)
    drop = np.array(drop)
    peak_times = librosa.frames_to_time(peak, sr=sr, hop_length=hop_length)
    drop_times = librosa.frames_to_time(drop, sr=sr, hop_length=hop_length)

    print("peak-Zeiten (s):", peak_times)
    print("drop-Zeiten (s):", drop_times)
    print("Anzahl der Lautstärkeänderungen:", len(peak_times) + len(drop_times))

    # -----------------------------------------------------
    # Plot erstellen
    # -----------------------------------------------------
    times = np.arange(len(y)) / sr
    plt.figure(figsize=(12, 4))
    plt.plot(times, y, label="Waveform")

    for tp in peak_times:
        plt.axvline(x=tp, color='red', linestyle='--', alpha=0.7)
    for tp in drop_times:
        plt.axvline(x=tp, color='green', linestyle='--', alpha=0.7)

    plt.xlabel("Zeit (s)")
    plt.ylabel("Amplitude")
    plt.title("Detektierte Lautstärkeänderungen")
    plt.legend()
    plt.tight_layout()

    # -------------------------------------
    # Output-Dateiname
    # -------------------------------------
    base = os.path.splitext(audio_file)[0]
    outpath = os.path.join(output_folder, f"{base}_lautstaerke.png")

    plt.savefig(outpath)
    plt.close()

    print("➡ Ergebnis gespeichert als:", outpath)


# -----------------------------------------------------
# Dateien im Input-Ordner finden
# -----------------------------------------------------
audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]

if not audio_files:
    print("Keine Audio-Dateien im Ordner gefunden!")
else:
    print("Gefundene Dateien:", audio_files)

# -----------------------------------------------------
# Hauptschleife
# -----------------------------------------------------
for filename in audio_files:
    try:
        analyse_audio_file(os.path.join(input_folder, filename))
    except Exception as e:
        print("⚠Fehler bei Datei:", filename)
        print(e)

print("\nFertig!")
