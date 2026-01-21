import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Ordner Einstellungen
# -----------------------------
input_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\Input_alles"
output_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\output"

os.makedirs(output_folder, exist_ok=True)

valid_ext = (".mp3", ".ogg", ".wav")

# ----------------------------------------------------------
# Analyse: Rauschen relativ zur Sprachlautstärke (SNR)
# ----------------------------------------------------------
def analyse_audio_file(audio_path):
    filename = os.path.basename(audio_path)
      # Audio laden
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Sicherheitscheck
    if len(y) < sr * 0.5:   # < 0.5 s
        return False

    # Kurzzeitenergie (RMS)
    hop_length = 512
    frame_length = 1024

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Robuste Dynamik
    low = np.percentile(rms, 10)
    high = np.percentile(rms, 90)

    # Schutz vor Division durch 0
    if low <= 1e-10:
        return True

    dynamic_range_db = 20 * np.log10(high / low)

    # Entscheidungsregel
    # < 20 dB → Rauschen dominiert
    # >= 20 dB → Sprache dominiert
    rauschen = dynamic_range_db < 20
    print(filename)
    print(rauschen)
    return rauschen


# ----------------------------------------------------------
# Ordnerdurchlauf
# ----------------------------------------------------------
audio_files = []
for subdir, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(valid_ext):
            audio_files.append(os.path.join(subdir, file))

if not audio_files:
    print("Keine Audio-Dateien gefunden!")

for f in audio_files:
    try:
        analyse_audio_file(f)
    except Exception as e:
        print("Fehler bei:", f)
        print(e)

print("\nFertig!")
