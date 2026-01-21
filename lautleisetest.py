#def analyse_lautstaerke(y, sr, audio_file):
#
#    # Frame-Parameter
#    frame_length = int(0.1 * sr)
#    hop_length = int(0.05 * sr)
#
#    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
#    rms_smooth = scipy.signal.medfilt(rms, kernel_size=7)
#
#    noise_th = 0.015
#    rms_no_noise = rms_smooth.copy()
#    rms_no_noise[rms_no_noise < noise_th] = 0
#
#    active = rms_no_noise[rms_no_noise > 0]
#    mean_amp = np.mean(active)
#
#    min_allowed = 0.1 * mean_amp
#    max_allowed = min(1.9 * mean_amp, 0.9)
#
#    threshold = otsu_threshold(rms_no_noise)
#
#    if threshold < min_allowed or threshold > max_allowed:
#        clipped = rms_no_noise[(rms_no_noise >= min_allowed) & (rms_no_noise <= max_allowed)]
#        threshold = otsu_threshold(clipped) if len(clipped) > 0 else mean_amp
#
#    # -------------------------
#    # Übergangserkennung
#    # -------------------------
#    margin_sec = 0.2 #gibt an wie viel davor und danach ton auftreten muss.
#    margin_frames = int(margin_sec / (hop_length / sr))
#
#    long_margin_sec = 1.0
#    long_frames = int(long_margin_sec / (hop_length / sr))
#
#    is_high = False
#    high_idx = []
#    low_idx = []
#
#    for idx in range(margin_frames, len(rms_smooth) - margin_frames):
#
#        vol = rms_smooth[idx]
#
#        local = rms_smooth[idx - margin_frames: idx + margin_frames + 1]
#        if np.any(local < noise_th):
#            is_high = (vol >= threshold)
#            continue
#
#        lf_start = max(0, idx - long_frames)
#        lf_end = min(len(rms_smooth), idx + long_frames)
#
#        before = rms_smooth[lf_start:idx]
#        after = rms_smooth[idx:lf_end]
#
#        before = before[before > noise_th]
#        after = after[after > noise_th]
#
#        if len(before) == 0 or len(after) == 0:
#            is_high = (vol >= threshold)
#            continue
#
#        diff = abs(np.mean(after) - np.mean(before))
#        if diff < 0.5 * mean_amp:
#            is_high = (vol >= threshold)
#            continue
#
#        if vol >= threshold:
#            if not is_high:
#                high_idx.append(idx)
#                is_high = True
#        else:
#            if is_high:
#                low_idx.append(idx)
#                is_high = False
#
#    high_times = librosa.frames_to_time(high_idx, sr=sr, hop_length=hop_length)
#    low_times = librosa.frames_to_time(low_idx, sr=sr, hop_length=hop_length)
#
#    #print("Otsu-Schwelle:", threshold)
#    #print("High-Übergänge:", high_times)
#    #print("Low-Übergänge:", low_times)
#    #print("Anzahl Übergänge:", len(high_times) + len(low_times))
#
#    times = librosa.frames_to_time(np.arange(len(rms_smooth)), sr=sr, hop_length=hop_length)
#    plt.figure(figsize=(12, 4))
#    plt.plot(times, rms_smooth, label="RMS")
#    plt.axhline(threshold, color='black', linestyle='--', label='Schwelle')
#
#    for t in high_times:
#        plt.axvline(t, color='red', linestyle='--')
#    for t in low_times:
#        plt.axvline(t, color='green', linestyle='--')
#
#    plt.legend()
#    plt.title("Lautstärkeänderungen (RMS + Otsu)")
#    plt.tight_layout()
#
#    out = os.path.join(output_folder, f"{audio_file}_lautstaerke3.png")
#    plt.savefig(out)
#    plt.close()
#    print("➡ Gespeichert:", out)
#
#    return [
#        {"name": "Lautstärke hoch", "times": high_times},
#        {"name": "Lautstärke runter", "times": low_times},
#    ]


#2
#def analyse_lautstaerke(y,sr,audio_file):
#    
#    # -------------------------
#    # Frame-Parameter
#    # -------------------------
#    frame_length = int(0.1 * sr)
#    hop_length = int(0.05 * sr)
#
#
#    rms = librosa.feature.rms(
#        y=y,
#        frame_length=frame_length,
#        hop_length=hop_length
#    )[0]
#
#    rms_smooth = scipy.signal.medfilt(rms, kernel_size=7)
#
#    active = rms_smooth[rms_smooth > 0]
#    mean_amp = np.mean(active) if len(active) > 0 else 0
#    
#
#
#    # -------------------------
#    # Otsu-Hilfsfunktion
#    # -------------------------
#    def otsu_in_range(data, low, high):
#        subset = data[(data >= low) & (data < high)]
#        if len(subset) < 10:
#            return None
#        return otsu_threshold(subset)
#
#    max_val=np.max(rms_smooth)if np.max(rms_smooth) > 0 else 1
#    max_allowed = min(1.9*np.mean(rms_smooth),0.9*max_val) if np.max(rms_smooth) > 0 else 0.9
#    
#
#    # -------------------------
#    # Schwellenbereiche
#    # -------------------------
#    p10 = 0.10 * max_val
#    p20 = 0.20 * max_val
#
#    # Noise-Schwelle (0–10 %)
#    noise_th = otsu_in_range(rms_smooth, 0.0, p10)
#    noise_th = noise_th if noise_th is not None else p10
#
#    # Übergang 1: noise_th – 20 %
#    th_low = otsu_in_range(rms_smooth, noise_th, p20)
#    th_low = th_low if th_low is not None else p20
#
#    # Übergang 2: 20 % – max
#    th_high = otsu_in_range(rms_smooth, p20, max_allowed)
#    th_high = th_high if th_high is not None else mean_amp
#
#    # -------------------------
#    # Übergangserkennung
#    # -------------------------
#    margin_sec = 0.2
#    margin_frames = int(margin_sec / (hop_length / sr))
#
#    long_margin_sec = 1.0
#    long_frames = int(long_margin_sec / (hop_length / sr))
#
#    state = 0  # 0=Stille, 1=leise/mittel, 2=laut
#    high_idx = []
#    low_idx = []
#
#    for idx in range(margin_frames, len(rms_smooth) - margin_frames):
#
#        vol = rms_smooth[idx]
#
#        # -------------------------
#        # Noise-Check in margin
#        # -------------------------
#        if vol < noise_th:
#            state = 0
#            continue
#
#        local = rms_smooth[idx - margin_frames: idx + margin_frames + 1]
#        if np.any(local < noise_th):
#            continue
#
#        # -------------------------
#        # Langzeit-Differenz
#        # -------------------------
#        lf_start = max(0, idx - long_frames)
#        lf_end = min(len(rms_smooth), idx + long_frames)
#
#        before = rms_smooth[lf_start:idx]
#        after = rms_smooth[idx:lf_end]
#
#        before = before[before >= noise_th]
#        after = after[after >= noise_th]
#
#        if len(before) == 0 or len(after) == 0:
#            if vol >= th_high:
#                state = 2
#            elif vol >= th_low:
#                state = 1
#            else:
#                state = 0
#            continue
#
#        diff = abs(np.mean(after) - np.mean(before))
#        if diff < 0.5 * mean_amp:
#            if vol >= th_high:
#                state = 2
#            elif vol >= th_low:
#                state = 1
#            else:
#                state = 0
#            continue
#
#        # -------------------------
#        # Zustandslogik (2 Schwellen)
#        # -------------------------
#        if vol >= th_high:
#            new_state = 2
#        elif vol >= th_low:
#            new_state = 1
#        else:
#            new_state = 0
#
#        if new_state > state:
#            high_idx.append(idx)
#        elif new_state < state:
#            low_idx.append(idx)
#
#        state = new_state
#
#    high_times = librosa.frames_to_time(
#        high_idx, sr=sr, hop_length=hop_length
#    )
#    low_times = librosa.frames_to_time(
#        low_idx, sr=sr, hop_length=hop_length
#    )
#
#    # -------------------------
#    # Plot
#    # -------------------------
#    times = librosa.frames_to_time(
#        np.arange(len(rms_smooth)), sr=sr, hop_length=hop_length
#    )
#
#    plt.figure(figsize=(12, 4))
#    plt.plot(times, rms_smooth, label="RMS")
#
#    plt.axhline(noise_th, color='gray', linestyle=':', label='Noise (0–10%)')
#    plt.axhline(th_low, color='blue', linestyle='--', label='Übergang 1 (Noise–20%)')
#    plt.axhline(th_high, color='purple', linestyle='--', label='Übergang 2 (20%–Max)')
#
#    for t in high_times:
#        plt.axvline(t, color='red', linestyle='--')
#    for t in low_times:
#        plt.axvline(t, color='green', linestyle='--')
#
#    plt.legend()
#    plt.title("Lautstärkeänderungen (RMS + 2 Übergangsschwellen)")
#    plt.tight_layout()
#
#    out = os.path.join(output_folder, f"{audio_file}_lautstaerke3.png")
#    plt.savefig(out)
#    plt.close()
#
#    print("➡ Gespeichert:", out)
#
#    return [
#        {"name": "Lautstärke hoch", "times": high_times},
#        {"name": "Lautstärke runter", "times": low_times},
#    ]

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

    # -----------------------------
    # Otsu-Schwelle
    # -----------------------------
def find_monotonic_region_before(slope, end_idx):
    """
    Sucht rückwärts ab end_idx den maximalen monotonen Bereich.
    Abbruch NUR bei Richtungswechsel.
    Mindestlänge = 1 Frame.
    """

    i = end_idx - 1
    if i < 0:
        return None, None

    direction = np.sign(slope[i])
    if direction == 0:
        return None, None

    start = i

    for j in range(i - 1, -1, -1):
        if np.sign(slope[j]) == direction:
            start = j
        else:
            break

    mean_region_slope = np.mean(slope[start:end_idx])

    return start, mean_region_slope


def analyse_audio_file(audio_file):
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    events = []
    # -------------------------
    # Frame-Parameter
    # -------------------------
    frame_length = int(0.05 * sr)
    hop_length = int(0.05 * sr)

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    rms_smooth = scipy.signal.medfilt(rms, kernel_size=7)

    times = librosa.frames_to_time(
        np.arange(len(rms_smooth)), sr=sr, hop_length=hop_length
    )

    # -------------------------
    # Parameter
    # -------------------------
    CONST_TOL = 0.01          #
    CONST_MIN_TIME = 0.3     # s
    SLOPE_FACTOR = 10
    LONG_TIME = 0.5          # s

    min_const_frames = int(CONST_MIN_TIME / (hop_length / sr))
    long_frames = int(LONG_TIME / (hop_length / sr))

    # -------------------------
    # Steigung
    # -------------------------
    slope = np.diff(rms_smooth)
    mean_slope = np.mean(np.abs(slope))

    high_idx = []
    low_idx = []

    i = 0
    while i < len(rms_smooth) - min_const_frames:

        segment = rms_smooth[i:i + min_const_frames]
        #mean_val = np.mean(segment)
        mean_val =rms_smooth[i]
        # -------------------------
        # Konstante Energie?
        # -------------------------
        if mean_val > 0:
            if np.all(np.abs(segment - mean_val) / mean_val <= CONST_TOL):
        

                start_idx = i
                end_idx = i + min_const_frames

                while end_idx < len(rms_smooth):
                    if abs(rms_smooth[end_idx] - mean_val) / mean_val <= CONST_TOL:
                        end_idx += 1
                    else:
                        break

                # -------------------------
                # gesamte monotone Phase davor
                # -------------------------
                mono_start, mono_slope = find_monotonic_region_before(slope, start_idx)

                if mono_start is not None:
                    if abs(mono_slope) > SLOPE_FACTOR * mean_slope:
                                    # -------------------------
                        # Langzeit-Differenz:
                        # VOR monotone Phase vs. NACH monotone Phase
                        # -------------------------
                        before_start = max(0, mono_start - long_frames)
                        before_end = mono_start

                        after_start = start_idx
                        after_end = min(len(rms_smooth), start_idx + long_frames)

                        before = rms_smooth[before_start:before_end]
                        after = rms_smooth[after_start:after_end]

                        if len(before) > 0 and len(after) > 0:
                            diff = abs(np.mean(after) - np.mean(before))

                            if diff > 0.5 * np.mean(rms_smooth):
                                if mono_slope > 0:
                                    high_idx.append(start_idx)
                                    ev_type = "high"
                                else:
                                    low_idx.append(start_idx)
                                    ev_type = "low"

                                events.append({
                                    "type": ev_type,
                                    "jump_idx": start_idx,

                                    # konstantes Niveau VORHER
                                    "before_start": before_start,
                                    "before_end": before_end,

                                    # monotone Steigungs-/Fallphase
                                    "mono_start": mono_start,
                                    "mono_end": start_idx,

                                    # konstantes Niveau NACHHER
                                    "after_start": after_start,
                                    "after_end": after_end,
                                })

                i = end_idx
            else:
                i += 1
        else:
            # mean_val == 0, also Segment ignorieren
            i += 1
            continue

    high_times = librosa.frames_to_time(high_idx, sr=sr, hop_length=hop_length)
    low_times = librosa.frames_to_time(low_idx, sr=sr, hop_length=hop_length)

    # =========================================================
    # Plot 1: RMS
    # =========================================================
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times, rms_smooth, color="black", linewidth=1.2, label="RMS")

    for ev in events:
        # konstantes Niveau vorher
        plt.axvspan(
            times[ev["before_start"]],
            times[ev["before_end"] - 1],
            color="blue",
            alpha=0.15,
            label="konstant vorher"
        )

        # monotone Phase
        plt.axvspan(
            times[ev["mono_start"]],
            times[ev["mono_end"] - 1],
            color="orange",
            alpha=0.25,
            label="monotone Phase"
        )

        # konstantes Niveau nachher
        plt.axvspan(
            times[ev["after_start"]],
            times[ev["after_end"] - 1],
            color="green",
            alpha=0.20,
            label="konstant nachher"
        )

        # Sprung
        plt.axvline(
            times[ev["jump_idx"]],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Sprung"
        )

    plt.title("RMS mit markierten Steigungs- und Konstantphasen")
    plt.ylabel("RMS")

    # doppelte Labels vermeiden
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
    # =========================================================
    # Plot 2: Audio-Signal
    # =========================================================
    plt.subplot(2, 1, 2)
    audio_times = np.arange(len(y)) / sr
    plt.plot(audio_times, y, color="gray", alpha=0.6, label="Audio-Signal")

    for ev in events:
        # konstantes Niveau vorher
        plt.axvspan(
            times[ev["before_start"]],
            times[ev["before_end"] - 1],
            color="blue",
            alpha=0.10
        )
        # monotone Phase
        plt.axvspan(
            times[ev["mono_start"]],
            times[ev["mono_end"] - 1],
            color="orange",
            alpha=0.15
        )
        # konstantes Niveau nachher
        plt.axvspan(
            times[ev["after_start"]],
            times[ev["after_end"] - 1],
            color="green",
            alpha=0.15
        )
        # Sprung-Zeitpunkt
        plt.axvline(
            times[ev["jump_idx"]],
            color="red",
            linestyle="--"
        )

    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.title("Audio-Signal mit erkannten Strukturphasen")
    out = os.path.join(output_folder, f"{os.path.basename(audio_file)}_lautstaerkeAnalyse.png")
    plt.savefig(out)
    plt.close()

    print("➡ Gespeichert:", out)
    # -------------------------
    # Ableitungen
    # -------------------------
    rms_deriv1 = np.gradient(rms_smooth)            # 1. Ableitung
    rms_deriv2 = np.gradient(rms_deriv1)            # 2. Ableitung

    # -------------------------
    # Zeitachsen
    # -------------------------
    times_rms = librosa.frames_to_time(np.arange(len(rms_smooth)), sr=sr, hop_length=hop_length)
    times_audio = np.arange(len(y)) / sr

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(14, 10))

    # Audio-Signal
    plt.subplot(3,1,1)
    plt.plot(times_audio, y, color='gray')
    plt.title("Audio-Signal")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")

    # RMS
    plt.subplot(3,1,2)
    plt.plot(times_rms, rms_smooth, color='black')
    plt.title("RMS")
    plt.xlabel("Zeit [s]")
    plt.ylabel("RMS")

    # Ableitungen
    plt.subplot(3,1,3)
    plt.plot(times_rms, rms_deriv1, label="1. Ableitung", color='blue')
    plt.plot(times_rms, rms_deriv2, label="2. Ableitung", color='red')
    plt.title("RMS Ableitungen")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Steigung")
    plt.legend()
    out = os.path.join(output_folder, f"{os.path.basename(audio_file)}_lAbleitungAnalyse.png")
    plt.savefig(out)
    plt.close()

    print("➡ Gespeichert:", out)

    return [
        {"name": "Lautstärke hoch", "times": high_times},
        {"name": "Lautstärke runter", "times": low_times},
    ]

def plot_audio_features(audio_path, output_folder):
        
    # --- Audio laden ---
    filename = os.path.basename(audio_path)
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    
    # --- STFT ---
    n_fft = 1024
    hop_length = 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    #soll hier noch was hin?



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
        plot_audio_features(os.path.join(input_folder, filename), output_folder)

    except Exception as e:
        print("Fehler bei Datei:", filename)
        print(e)

print("\nFertig!")
