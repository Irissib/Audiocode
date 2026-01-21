import librosa
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------
# Ordner
# -----------------------------------------------------
#entweder:
#input_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\Input"
#output_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\output"

#oder:
input_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\Input_alles"
output_folder = r"C:\Users\irisb\Desktop\Iris\hiwi\audiocode\output"

os.makedirs(output_folder, exist_ok=True)

# -----------------------------------------------------
# Erlaubte Dateiendungen
# -----------------------------------------------------
valid_ext = (".mp3", ".ogg", ".wav")

# -----------------------------------------------------
# Analyse-Funktion
# -----------------------------------------------------
def analyse_audio_file(filepath):
    is_high = False
    audio_file = os.path.basename(filepath)

    print("\nVerarbeite:", audio_file)

    # Audio laden
    y, sr = librosa.load(filepath, sr=None, mono=True)

    #Parameter
    frame_length = int(0.1 * sr)
    hop_length = int(0.05 * sr)

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    max_abs_per_frame = np.max(np.abs(frames), axis=0)#maximum pro amplitude
    max_abs = np.max(np.abs(y), axis=0)


    volume=True #Hier wird überprüft ob das Audio genug Lautstärke hat
    if max_abs == 0:
        print("Datei ohne Ton")
        volume=False
    if max_abs < 0.1 and max_abs != 0:
        print("Datei allgemein sehr leise")
        volume=False


    # Rauschfilter
    threshold_noise = 0.01 #Alles darunter soll als Rauschen wargenommen werden.
    clean_volumes = max_abs_per_frame.copy()
    clean_volumes[clean_volumes < threshold_noise] = 0# Rauschen auf 0
    active_volumes = clean_volumes[clean_volumes > 0]# Nur Audio ohne Rauschen

    # Otsu-Schwelle berechnen. Das Audio wird durch diese SChwellen in zwei gruppen geteilt. leise und LAut, mit möglichst großen unterschieden zwischen den 2 Gruppen

    mean_amp = np.mean(active_volumes)# Durchschnitts Lautstärke des Audios ohne Rauschen
    min_allowed =0.1 * mean_amp#sichert das die Otsu-schwelle nicht zu niedrig oder zu hoch ist
    max_allowed = min(1.9 * mean_amp,0.9) 

    def otsu_threshold(volumes):
        hist, bin_edges = np.histogram(volumes, bins=256, range=(0, 1))
        total = volumes.size
        sum_total = np.sum(bin_edges[:-1] * hist)
        sumB = 0.0
        wB = 0
        max_var = 0.0
        threshold = 0.0
        for i in range(256):
            wB += hist[i]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += bin_edges[i] * hist[i]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            var_between = wB * wF * (mB - mF) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = bin_edges[i]
        return threshold
    threshold = otsu_threshold(clean_volumes)
    if threshold < min_allowed or threshold > max_allowed:
    # nur Werte innerhalb des erlaubten Bereichs auswählen
        clipped_volumes = clean_volumes[(clean_volumes >= min_allowed) & (clean_volumes <= max_allowed)]#Fals Nötig wird das Audio auf die Bereiche zwischen min und max reduziert um eine Schwelle in diesem Bereich zu erzwingen.
        if len(clipped_volumes) > 0:
            threshold = otsu_threshold(clipped_volumes)
        else:
            # falls keine Werte im Bereich, auf Mittelwert setzen
            threshold = mean_amp


   # -----------------------------
    # Margin & Übergangszählung über max_abs_per_frame
    # -----------------------------
    over_thresh  = max_abs_per_frame[max_abs_per_frame >= threshold]#Gruppe:laut
    under_thresh = max_abs_per_frame[max_abs_per_frame < threshold]#Gruppe:Leise
    margin_sec =0.2 #gibt an wie viel davor und danach ton auftreten muss.
    margin_frames = int(margin_sec / (hop_length / sr))

    high = []
    low = []


    # gültige Indizes, die genügend Abstand zu Rand haben
    valid_idx = np.arange(margin_frames, len(max_abs_per_frame) - margin_frames)#angepasste länge
    if (len(over_thresh) == 0 or len(under_thresh) == 0 or np.mean(over_thresh)-np.mean(under_thresh)) < 0.4 * max_abs or volume is False: # keine Übergänge, Schwelle nicht legitim weil die driffernz der gruppe groß und klein zu gering. 
        pass
    else:
        for idx in valid_idx:
            # Umgebung prüfen: margin_frames vor und nach dem Frame
            start = idx - margin_frames
            end = idx + margin_frames + 1  # +1, weil Python slice exklusiv

            local_max = max_abs_per_frame[start:end]#Vektor mit allen maximas der Frames
            vol = max_abs_per_frame[idx]
            # Übergang nur, wenn alle Frames > threshold_noise,Rauschen
            if np.any(local_max < threshold_noise):
                if vol >= threshold:
                    is_high = True
                else:
                    is_high = False
                continue  # Rauschen in Umgebung → Übergang ignorieren

            

            # -----------------------------------------------------------
            # Langzeit-Lautstärkeprüfung
            # -----------------------------------------------------------
            long_margin_sec = 1
            long_frames = int(long_margin_sec / (hop_length / sr))

            lf_start = max(0, idx - long_frames)
            lf_end   = min(len(max_abs_per_frame), idx + long_frames + 1)

            before = max_abs_per_frame[lf_start:idx]
            after  = max_abs_per_frame[idx:lf_end]

            if len(before) == 0 or len(after) == 0:
                if vol >= threshold:#Wieder das, damit die aktuelle Lautstärke immer aktuell ist
                    is_high = True
                else:
                    is_high = False
                continue
            before = before[before > 0.015]
            after = after[after > 0.015]
            avg_before = np.mean(before) if len(before) > 0 else 0 #Durschnittswerte 1 sec vor und nach der erkannten schwelle
            avg_after = np.mean(after) if len(after) > 0 else 0


            diff = abs(avg_after - avg_before)

            min_required_change = 0.5 * np.mean(active_volumes)

            if diff < min_required_change:
                if vol >= threshold:#Wieder das, damit die aktuelle Lautstärke immer aktuell ist
                    is_high = True
                else:
                    is_high = False
                continue

            # Normale Übergangszählung
            vol = max_abs_per_frame[idx]
            if vol >= threshold:
                if not is_high:#übergang zu lauter gruppe, wenn vorher nicht laut. is_high
                    high.append(idx)
                    is_high = True
            else:
                if is_high:
                    low.append(idx)
                    is_high = False
        # Ergebnisse
    high = np.array(high)
    low = np.array(low)
    high_times = librosa.frames_to_time(high, sr=sr, hop_length=hop_length)
    low_times = librosa.frames_to_time(low, sr=sr, hop_length=hop_length)

    print('grenze:',threshold)
    print("high-Zeiten (s):", high_times)
    print("low-Zeiten (s):", low_times)
    print("Anzahl der Lautstärkeänderungen:", len(high_times) + len(low_times))

    # -----------------------------------------------------
    # Plot erstellen
    # -----------------------------------------------------
    times = np.arange(len(y)) / sr
    plt.figure(figsize=(12, 4))
    plt.plot(times, y, label="Waveform")  
    for tp in high_times:
        plt.axvline(x=tp, color='red', linestyle='--', alpha=0.7)
    for tp in low_times:
        plt.axvline(x=tp, color='green', linestyle='--', alpha=0.7) 
    plt.xlabel("Zeit (s)")
    plt.ylabel("Amplitude")
    plt.title("Detektierte Lautstärkeänderungen")
    plt.legend()
    plt.tight_layout()  # -------------------------------------
    # Output-Dateiname
    # -------------------------------------
    base = os.path.splitext(audio_file)[0]
    outpath = os.path.join(output_folder, f"{base}_2lautstaerke.png")   
    plt.savefig(outpath)
    plt.close() 
    print("➡ Ergebnis gespeichert als:", outpath)


# -----------------------------------------------------
# Dateien im Input-Ordner finden
# -----------------------------------------------------
#entweder:
# audio_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)])


# if not audio_files:
#     print("Keine Audio-Dateien im Ordner gefunden!")
# else:
#     print("Gefundene Dateien:", audio_files)

# # -----------------------------------------------------
# # Hauptschleife
# # -----------------------------------------------------



# for filename in audio_files:
#     try:
#         analyse_audio_file(os.path.join(input_folder, filename))
#     except Exception as e:
#         print("Fehler bei Datei:", filename)
#         print(e)

# print("\nFertig!")


#oder:
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
        analyse_audio_file(os.path.join(input_folder, filename))
    except Exception as e:
        print("Fehler bei Datei:", filename)
        print(e)

print("\nFertig!")