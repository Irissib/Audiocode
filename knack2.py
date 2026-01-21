import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal

# -----------------------------
# Ordner
# -----------------------------
input_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\Input_alles"
output_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\output"

os.makedirs(output_folder, exist_ok=True)

valid_ext = (".mp3", ".wav", ".ogg")

# Liste aller gültigen Dateien
audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]

files_with_knacks2 = []   # Übersichtsliste


# ---------------------------------------------------------
# FUNKTION: Analyse Knacktyp 2 für EINE Datei
# ---------------------------------------------------------
def analyse_knacktyp2(filepath):

    filename = os.path.basename(filepath)

    
    # -----------------------------
    # Audio laden
    # -----------------------------
    y, sr = librosa.load(filepath, sr=None, mono=True)

    # Hochpassfilter bei 2 kHz
    cutoff = 10000  # Hz
    b, a = scipy.signal.butter(1, cutoff/(sr/2), 'high')#hochpassfilter
    y_hp = scipy.signal.filtfilt(b, a, y)
    diff = np.abs(np.diff(y_hp))#ableitung
    noise=0.15*max(y)

    knacks2 = []

    block_size = 500
    window_size = 1000
    thresholds = np.zeros(len(diff))
    little_thresholds = np.zeros(len(diff))

    # Schritt 1: Thresholds blockweise berechnen
    for start in range(0, len(diff), block_size):
        end = min(len(diff), start + block_size)
        w_start = max(0, start - window_size//2)
        w_end   = min(len(diff), end + window_size//2)
        
        local_median = np.median(diff[w_start:w_end])
        local_max    = np.max(diff[w_start:w_end])
        
        thresholds[start:end] = 50 * local_median
        little_thresholds[start:end] = 0.25 * local_max

    # Schritt 2: Knacks-Erkennung pro Sample
    for i in range(50, len(diff)-50):
        if diff[i] > thresholds[i] and diff[i] > little_thresholds[i]:
            if diff[i-1] < thresholds[i-1] and diff[i+1] < thresholds[i+1]:
                if np.sign(y_hp[i-1]) != np.sign(y_hp[i]) or np.sign(y_hp[i]) != np.sign(y_hp[i+1]):
                    if np.max(np.abs(y[i-50:i+51])) > noise:
                        knacks2.append(i)

    knacks2 = np.array(knacks2)
    times_selected = np.round(knacks2 / sr, 4)

    anzahl_knacks2 = len(times_selected)

    print(filename)
    print("Knacks-Zeiten (s):", times_selected)
    print("Anzahl der Knacksen:", anzahl_knacks2)

    

    # Falls Knacks gefunden → in Übersicht eintragen
    if anzahl_knacks2 > 0:
        files_with_knacks2.append(filename)

    # -----------------------------
    # Visualisierung
    # -----------------------------
    times = np.arange(len(y)) / sr

    plt.figure(figsize=(12, 4))
    plt.plot(times, y, label="Waveform")

    for t in times_selected:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.7)

    plt.title("Detektierte Knackser Typ 2")
    plt.xlabel("Zeit (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    outname = os.path.join(
        output_folder,
        filename.rsplit(".", 1)[0] + "_knacktyp2.png"
    )
    plt.savefig(outname)
    plt.close()


# ---------------------------------------------------------
# HAUPTSCHLEIFE: Alle Dateien verarbeiten
# ---------------------------------------------------------
print("Gefundene Dateien:", audio_files)

audio_files_2 = []
for subdir, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(valid_ext):
            audio_files_2.append(os.path.join(subdir, file))

if not audio_files_2:
    print("Keine Audio-Dateien im Ordner gefunden!")
else:
    print("Gefundene Dateien:", audio_files_2)

for filename in audio_files_2:
    try:
        analyse_knacktyp2(os.path.join(input_folder, filename))
    except Exception as e:
        print("Fehler bei Datei:", filename)
        print(e)

print("\nFertig!")


# ---------------------------------------------------------
# ALPHABETISCH SORTIERTE ÜBERSICHT
# ---------------------------------------------------------
print("\n--------------------------------")
print("ÜBERSICHT – Dateien mit mindestens einem Knacktyp 2:")

if files_with_knacks2:
    for f in sorted(files_with_knacks2):
        print(" -", f)
else:
    print("Keine Knacksen Typ 2 gefunden.")

print("--------------------------------")
print("Fertig!")
