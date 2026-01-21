import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Ordner Einstellungen
# -----------------------------
input_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\Input"
output_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\output"
sr_target = 48000                  # Ziel-Samplingrate spsäter überschrieben

# Falls der Zielordner nicht existiert → anlegen
os.makedirs(output_folder, exist_ok=True)

# Unterstützte Endungen
valid_ext = (".mp3", ".ogg", ".wav")

# ----------------------------------------------------------
# Funktion zur Übersteuerungs-Analyse
# ----------------------------------------------------------
def analyse_audio_file(audio_path):

    print("\nVerarbeite:", audio_path)

    # Audio laden
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Parameter
    frame_length = int(0.05 * sr)
    hop_length = int(0.05 * sr)
    
    max_amp = np.max(np.abs(y))
    if max_amp >= 1.0:              #nach Wahrer Übersteuerung Prüfen. Audios übersteuert meist aber bei 0.95 und erreichen 1 nicht
        high_thresh = 1.0
        low_thresh = 0.98
    else:
        high_thresh = 0.95
        low_thresh = 0.93
        

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    max_abs_per_frame = np.max(np.abs(frames), axis=0)#maximum pro Frame

    ueber = []
    ueber_end = []
    high = False

    for i in range(0, len(max_abs_per_frame)):
        if all(max_abs_per_frame[i:i+3] >= high_thresh) and high is False:#wenn hohe Werte und davor niedrig:Anfang übersteuerung
            ueber.append(i)
            high = True
        if max_abs_per_frame[i] < low_thresh and high is True:#Wenn niedriger Werte und davor hoch:Ende Übersteuerung
            high = False
            ueber_end.append(i)
    #Übersteuerung-->zeit der Übersteuerung
    ueber = np.array(ueber)
    ueber_end = np.array(ueber_end)
    ueber_times = librosa.frames_to_time(ueber, sr=sr, hop_length=hop_length)
    ueber_end_times = librosa.frames_to_time(ueber_end, sr=sr, hop_length=hop_length)

    print("Übersteuerungs-Beginn Zeiten (s):", ueber_times)
    print("Anzahl Übersteuerungen:", len(ueber_times))

    # Visualisierung
    times = np.arange(len(y)) / sr
    plt.figure(figsize=(12, 4))
    plt.plot(times, y, label="Waveform", color='blue')
    for tp in ueber_times:
        plt.axvline(x=tp, color='red', linestyle='--', alpha=0.7)
    for tp in ueber_end_times:
        plt.axvline(x=tp, color='green', linestyle='--', alpha=0.7)

    plt.xlabel("Zeit (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Detektierte Übersteuerung: {os.path.basename(audio_path)}")
    plt.legend()
    plt.tight_layout()

    # Datei-spezifischer Output Name
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(output_folder, f"{base_name}_uebersteuerung.png")

    plt.savefig(out_path)
    plt.close()

    print("→ gespeichert als:", out_path)


# ----------------------------------------------------------
# Alle Dateien im Ordner durchgehen
# ----------------------------------------------------------
audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]

if not audio_files:
    print("Keine Audio-Dateien im Ordner gefunden!")
else:
    print("Gefundene Dateien:", audio_files)

for filename in audio_files:
    try:
        analyse_audio_file(os.path.join(input_folder, filename))
    except Exception as e:
        print("⚠Fehler bei Datei:", filename)
        print(e)

print("\nFertig!")
