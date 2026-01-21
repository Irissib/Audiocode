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
#--------------------------
#Analysefunktion-----------------------
#---------------------------------
def otsu_threshold(vols):
    vols = np.clip(vols, 0, 1)
    hist, bins = np.histogram(vols, bins=256, range=(0, 1))
    total = vols.size
    sum_total = np.sum(bins[:-1] * hist)

    sumB = 0
    wB = 0
    max_var = 0
    threshold = 0

    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break

        sumB += bins[i] * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = bins[i]

    return threshold
def analyse_mit_otsu(audio_file):
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    filename=os.path.basename(audio_file)

    # Frame-Parameter
    frame_length = int(0.1 * sr)
    hop_length = int(0.05 * sr)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_smooth = scipy.signal.medfilt(rms, kernel_size=7)

    noise_th = 0.015
    rms_no_noise = rms_smooth.copy()
    rms_no_noise[rms_no_noise < noise_th] = 0

    active = rms_no_noise[rms_no_noise > 0]
    mean_amp = np.mean(active)

    min_allowed = 0.1 * mean_amp
    max_allowed = min(1.9 * mean_amp, 0.9)

    threshold = otsu_threshold(rms_no_noise)

    if threshold < min_allowed or threshold > max_allowed:
        clipped = rms_no_noise[(rms_no_noise >= min_allowed) & (rms_no_noise <= max_allowed)]
        threshold = otsu_threshold(clipped) if len(clipped) > 0 else mean_amp

    # -------------------------
    # Übergangserkennung
    # -------------------------
    margin_sec = 0.2 #gibt an wie viel davor und danach ton auftreten muss.
    margin_frames = int(margin_sec / (hop_length / sr))

    long_margin_sec = 1.0
    long_frames = int(long_margin_sec / (hop_length / sr))

    is_high = False
    low_idx = []

    for idx in range(margin_frames, len(rms_smooth) - margin_frames):

        vol = rms_smooth[idx]

        local = rms_smooth[idx - margin_frames: idx + margin_frames + 1]
        if np.any(local < noise_th):
            is_high = (vol >= threshold)
            continue

        lf_start = max(0, idx - long_frames)
        lf_end = min(len(rms_smooth), idx + long_frames)

        before = rms_smooth[lf_start:idx]
        after = rms_smooth[idx:lf_end]

        before = before[before > noise_th]
        after = after[after > noise_th]

        if len(before) == 0 or len(after) == 0:
            is_high = (vol >= threshold)
            continue

        diff = abs(np.mean(after) - np.mean(before))
        if diff < 0.5 * mean_amp:
            is_high = (vol >= threshold)
            continue

        if vol <= threshold:
            if is_high:
                low_idx.append(idx)
                is_high = False

    
    low_times = librosa.frames_to_time(low_idx, sr=sr, hop_length=hop_length)

    #print("Otsu-Schwelle:", threshold)
    #print("High-Übergänge:", high_times)
    #print("Low-Übergänge:", low_times)
    #print("Anzahl Übergänge:", len(high_times) + len(low_times))

    times = librosa.frames_to_time(np.arange(len(y)), sr=sr, hop_length=hop_length)
    plt.figure(figsize=(12, 4))
    plt.plot(times, y, label="Otsu")
    plt.axhline(threshold, color='black', linestyle='--', label='Schwelle')


    for t in low_times:
        plt.axvline(t, color='green', linestyle='--')

    plt.legend()
    plt.title("Lautstärkeänderungen (RMS + Otsu)")
    plt.tight_layout()

    out = os.path.join(output_folder, f"{filename}_lautstaerke3.png")
    plt.savefig(out)
    plt.close()
    print("➡ Gespeichert:", out)

    return [
        {"name": "Lautstärke runter", "times": low_times},
    ]

def analyse_mit_ableitung(audio_file):
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    filename=os.path.basename(audio_file)

    # Frame-Parameter
    frame_length = int(0.1 * sr)
    hop_length = int(0.05 * sr)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_smooth = scipy.signal.medfilt(rms, kernel_size=7)
     # -------------------------
    # Ableitungen
    # -------------------------
    rms_deriv1 = np.gradient(rms_smooth)       # 1. Ableitung
    rms_deriv2 = np.gradient(rms_deriv1)       # 2. Ableitung

    # -------------------------
    # Parameter für Low-Events
    # -------------------------
    global_min = np.min(rms_deriv2)
    min_thresh = 0.1 * abs(global_min)
    zero_tol = 1e-4  # Schwellenwert für "nahe 0" zwischen Minima

    # -------------------------
    # Lokale Minima der 2. Ableitung
    # -------------------------
    local_min_idx = (np.r_[True, rms_deriv2[1:-1] < rms_deriv2[:-2]] &
                     np.r_[True, rms_deriv2[1:-1] < rms_deriv2[2:]])  # bool array
    local_min_idx = np.where(local_min_idx)[0]

    # nur signifikante Minima
    significant_min_idx = local_min_idx[rms_deriv2[local_min_idx] <= -min_thresh]

    # -------------------------
    # Suche nach aufeinanderfolgenden Minima
    # -------------------------
    low_times = []
    i = 0
    while i < len(significant_min_idx) - 1:
        idx1 = significant_min_idx[i]
        idx2 = significant_min_idx[i + 1]

        # Werte zwischen den Minima
        if idx2 > idx1 + 1:
            deriv1_between = rms_deriv1[idx1+1:idx2]
            deriv2_between = rms_deriv2[idx1+1:idx2]

            # prüfen, ob Ableitungen dazwischen ~0 sind
            if np.all(np.abs(deriv1_between) < zero_tol) and np.all(np.abs(deriv2_between) < zero_tol):
                # zwei Minima mit Lücke und ruhiger Zone
                low_times.append((idx1 + idx2) / 2)  # Mittelwert der Indizes
                i += 2
                continue

        else:
            # verschmolzene Minima (direkt nebeneinander)
            low_times.append(idx1)
            i += 2
            continue

        i += 1
    # -------------------------
    # Plot (optional)
    # -------------------------
    times = librosa.frames_to_time(np.arange(len(rms_smooth)), sr=sr, hop_length=hop_length)
    plt.figure(figsize=(12, 6))
    plt.plot(times,rms_smooth, label="RMS")
    plt.plot(times, rms_deriv1, label="1. Ableitung")
    plt.plot(times, rms_deriv2, label="2. Ableitung")
    for t in low_times:
        plt.axvline(t, color='red', linestyle='--', label="Low-Event")
    plt.legend()
    plt.title(f"Analyse 2. Ableitung: {filename}")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude / Ableitungen")
    out = os.path.join(output_folder, f"{filename}_lautstaerke3.png")
    plt.savefig(out)
    plt.close()
    print("➡ Gespeichert:", out)

    return [
        {"name": "Lautstärke runter", "times": low_times},
    ]
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
        analyse_mit_otsu(os.path.join(input_folder, filename))
        analyse_mit_ableitung(os.path.join(input_folder, filename))

    except Exception as e:
        print("Fehler bei Datei:", filename)
        print(e)

print("\nFertig!")