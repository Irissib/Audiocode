import librosa
import numpy as np
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
# Liste für Dateien mit Knacksen
# -----------------------------------------------------
files_with_knacks = []

# -----------------------------------------------------
# Analysefunktion für "Knacktyp 1"
# -----------------------------------------------------
def analyse_audio_file(filepath):

    audio_file = os.path.basename(filepath)
    print("\nVerarbeite:", audio_file)

    # -----------------------------
    # Audio laden
    # -----------------------------
    sr_target = 48000
    y, sr = librosa.load(filepath, sr=None)
    sr_target= sr

    # -----------------------------
    # Parameter
    # -----------------------------
    frame_length = int(0.005 * sr)
    hop_length   = int(0.003 * sr)

    max_amp = np.max(np.abs(y))
    low_thresh  = 0.01 * max_amp   # 1%
    high_thresh = 0.05 * max_amp   # 5%

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    max_abs_per_frame = np.max(np.abs(frames), axis=0)

    indices = []

    # -----------------------------
    # Knack-Erkennung
    # -----------------------------
    for i in range(3, len(max_abs_per_frame) - 3):
        current = max_abs_per_frame[i]

        if current <= low_thresh:#Amplitude muss klein sein im Knack Bereich
            prev = max_abs_per_frame[i-3:i]
            next_ = max_abs_per_frame[i+1:i+4]

            if np.max(prev) >= high_thresh and np.max(next_) >= high_thresh:#Amplitude muss irgendwo in näherer umgebung 3*frame groß sein.
                indices.append(i)
    #knackse -->zeit der knackse
    indices = np.array(indices)
    times_selected = librosa.frames_to_time(indices, sr=sr, hop_length=hop_length)
    anzahl_knacks = len(indices)

    # -----------------------------
    # Übersichtsliste befüllen
    # -----------------------------
    if anzahl_knacks > 0:
        files_with_knacks.append(audio_file)

    # -----------------------------
    # Ausgabe
    # -----------------------------
    print("Knacks-Zeiten (s):", times_selected)
    print("Anzahl der Knacksen:", anzahl_knacks)

    # -----------------------------
    # Visualisierung
    # -----------------------------
    times = np.arange(len(y)) / sr
    plt.figure(figsize=(12,4))
    plt.plot(times, y, label="Waveform", color='blue')

    for tp in times_selected:
        plt.axvline(x=tp, color='red', linestyle='--', alpha=0.7)

    plt.xlabel("Zeit (s)")
    plt.ylabel("Amplitude")
    plt.title("Detektierte Knacksen – Typ 1")
    plt.legend()
    plt.tight_layout()

    # -----------------------------
    # Output-Datei speichern
    # -----------------------------
    base = os.path.splitext(audio_file)[0]
    outpath = os.path.join(output_folder, f"{base}_knacktyp1.png")

    plt.savefig(outpath)
    plt.close()

    print("➡ Ergebnis gespeichert als:", outpath)


# -----------------------------------------------------
# Dateien im Ordner suchen
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
        print("⚠ Fehler bei Datei:", filename)
        print(e)

# -----------------------------------------------------
# Übersicht ausgeben
# -----------------------------------------------------
print("\n--------------------------------")
print("ÜBERSICHT – Dateien mit mindestens einem Knack:")
if files_with_knacks:
    for f in sorted(files_with_knacks):   # hier sortieren
        print(" -", f)
else:
    print("Keine Knacksen gefunden.")
print("--------------------------------")