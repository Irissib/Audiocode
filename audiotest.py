import librosa
import numpy as np
import scipy.signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.cluster import KMeans
#Noch zu tun: evt audios korrigieren(lauteleise)(rauschbarriere noise_th genauer. dann ist die marge in der hintischt besser)
#maybe lange 0.3s knackser
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

# Ergebnis tabelle
summary_results = []

# -----------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------

def load_audio(filepath):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    return y, sr


def check_no_signal(y, sr):

    events = []

    max_abs = np.max(np.abs(y))
    mean_amp = np.mean(np.abs(y))

    if max_abs == 0:
        events.append({"name": "Kein Ton", "times": []})
        return True, events #ermöglischt nachher die übersichtstabelle durch events, Treu ermöglichst skippen von weiteren Analysen

    if mean_amp < 0.01:
        events.append({"name": "Sehr leise", "times": []})

    return False, events



# -----------------------------
# Otsu-Schwelle
# Teilt audio in 2 Gruppen mit möglichst großer schwelle
# -----------------------------
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
#--------------------------
#Analysefunktion-----------------------
#----------------------------------
def analyse_lautstaerke(y, sr, audio_file):

    # Frame-Parameter
    frame_length = int(0.1 * sr)
    hop_length = int(0.05 * sr)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_smooth = scipy.signal.medfilt(rms, kernel_size=7)

    noise_sec=0.2
    noise_frames = int(noise_sec * sr / hop_length)

    noise_th=max(2*np.max(rms_smooth[:noise_frames]),0.05*np.max(rms_smooth))  # Rauschreferenz

    rms_no_noise = rms_smooth.copy()
    rms_no_noise[rms_no_noise < noise_th] = 0

    active = rms_no_noise[rms_no_noise > 0]
    mean_amp = np.mean(active)
    max_allowed = min(0.9 * np.max(active), 0.9)
    active_usage = active[active < max_allowed]

    th_low=otsu_threshold(active_usage)
    


   #    # -------------------------
    # Übergangserkennung
    # -------------------------
    margin_sec = 0.3
    margin_frames = int(margin_sec / (hop_length / sr)) 
#
    long_margin_sec = 1
    long_frames = int(long_margin_sec / (hop_length / sr ))
#
    is_high = False #false wenn unter Schwelle, true wenn über schwelle th_low
    low_idx =  []
#
    for idx in range(margin_frames, len(rms_smooth) - margin_frames) :
#
        vol = rms_smooth[idx]
#
        # -------------------------
        # Noise-Check in margin
        # -------------------------
        
#
        local = rms_smooth[idx - margin_frames: idx + margin_frames + 1]

        if np.any(local < noise_th):
            is_high = vol>th_low
            continue
#
        # -------------------------
        # Langzeit-Differenz
        # -------------------------
        lf_start = max(0, idx - long_frames)
        lf_end = min(len(rms_smooth), idx + long_frames)
#
        before = rms_smooth[lf_start:idx]
        after = rms_smooth[idx:lf_end ]
#
        before = before[before >= noise_th]
        after = after[after >= noise_th ]
#
        if len(before) == 0 or len(after) == 0:
            is_high = vol>th_low
            continue
#
        diff = abs(np.mean(after) - np.mean(before))
        if diff < 0.5 * mean_amp:
            is_high = vol>th_low
            continue
#
        # -------------------------
        # Zustandslogik (2 Schwellen)
        # -------------------------
       
        new_is_high = vol>th_low
        
#
        if (not new_is_high) and is_high:
            low_idx.append(idx)
            is_high = new_is_high 
        else:
            is_high = new_is_high   
#
    
    low_times = librosa.frames_to_time(
        low_idx, sr=sr, hop_length=hop_length)
    
#   #Prüfen nach schnelleren sprüngen
    fast_low_idx =  []
       # 1. Erste Ableitung der geglätteten RMS
    drms = np.diff(rms_smooth)

    if len(drms) > 0:
        d_min = np.min(drms)  # stärkster negativer Abfall

        # Nur wenn es überhaupt negative Abfälle gibt
        if d_min < 0:
            th_drop = 0.§ * d_min  # 10 % des negativsten Wertes

            for i, d in enumerate(drms):
                if d <= th_drop:
                    idx = i + 1  # diff → RMS-Index

                    fast_low_idx.append(idx)

   # -------------------------
    # Plot: RMS + Ableitung
    # -------------------------

    times = librosa.frames_to_time(
        np.arange(len(rms_smooth)), sr=sr, hop_length=hop_length)

    drms = np.diff(rms_smooth)
    times_d = times[1:]  # diff ist 1 kürzer

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # ---- RMS ----
    ax1.plot(times, rms_smooth, label="RMS", color="black")
    ax1.axhline(noise_th, color="gray", linestyle=":", label="Noise")
    ax1.axhline(th_low, color="blue", linestyle="--", label="Otsu-Schwelle")

    for t in low_times:
        ax1.axvline(t, color="green", linestyle="--", alpha=0.7)

    for i in fast_low_idx:
        ax1.axvline(times[i], color="orange", linestyle=":", alpha=0.7)

    ax1.set_ylabel("RMS")
    ax1.legend(loc="upper right")

    # ---- Ableitung ----
    ax2.plot(times_d, drms, label="d(RMS)/dt", color="purple")
    ax2.axhline(0, color="black", linewidth=0.8)

    # Markiere schnelle Abfälle
    for i in fast_low_idx:
        ax2.axvline(times[i], color="orange", linestyle=":", alpha=0.7)

    ax2.set_ylabel("Δ RMS")
    ax2.set_xlabel("Zeit [s]")
    ax2.legend(loc="upper right")

    plt.suptitle("Lautstärkeanalyse: RMS & schnelle Abfälle")
    plt.tight_layout()
    #
    out = os.path.join(output_folder, f"{audio_file}_lautstaerkeotsu.png")
    plt.savefig(out)
    plt.close ()
#
    print("➡ Gespeichert:", out) 

    



    
    
#
    return [
        {"name": "Lautstärke runter", "times": low_times}
    ]




def analyse_knacks(y, sr, audio_file):

    frame_length = int(0.005 * sr)
    hop_length   = int(0.003 * sr)

    max_amp = np.max(np.abs(y))
    low_thresh  = 0.01 * max_amp
    high_thresh = 0.05 * max_amp

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    max_abs_per_frame = np.max(np.abs(frames), axis=0)

    indices = []

    for i in range(3, len(max_abs_per_frame) - 3):
        current = max_abs_per_frame[i]

        if current <= low_thresh:
            prev = max_abs_per_frame[i-3:i]
            next_ = max_abs_per_frame[i+1:i+4]

            if np.max(prev) >= high_thresh and np.max(next_) >= high_thresh:
                indices.append(i)

    times_selected = librosa.frames_to_time(indices, sr=sr, hop_length=hop_length)

    # Knacks-Zeiten zusammenfassen, wenn <1ms auseinander
    # -----------------------------
    if len(times_selected) > 1:
        grouped = []
        buffer = [times_selected[0]]

        for t in times_selected[1:]:
            if (t - buffer[-1]) <= 0.01:  # 1 ms Abstand
                buffer.append(t)
            else:
                # Mittelwert der Gruppe
                grouped.append(np.mean(buffer))
                buffer = [t]

        # letzte Gruppe
        grouped.append(np.mean(buffer))
        times_selected = np.array(grouped)

    #print("Knacks-Zeiten (s):", times_selected)
    #print("Anzahl der Knacksen:", len(indices))
    return [
        {"name": "Knacks", "times": times_selected}
    ]

def analyse_knacks2(y, sr, audio_file):

    # Hochpassfilter bei 2 kHz
    cutoff = 10000  # Hz
    nyquist = sr / 2
    cutoff = min(cutoff, nyquist*0.99)  # nyquist erfüllend
    b, a = scipy.signal.butter(1, cutoff/nyquist , 'high')#hochpassfilter
    y_hp = scipy.signal.filtfilt(b, a, y)
    diff = np.abs(np.diff(y_hp))#ableitung
    noise=0.15*max(np.abs(y))

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

    # Knacks-Zeiten zusammenfassen, wenn <1ms auseinander
    # -----------------------------
    if len(times_selected) > 1:
        grouped = []
        buffer = [times_selected[0]]

        for t in times_selected[1:]:
            if (t - buffer[-1]) <= 0.01:  # 1 ms Abstand
                buffer.append(t)
            else:
                # Mittelwert der Gruppe
                grouped.append(np.mean(buffer))
                buffer = [t]

        # letzte Gruppe
        grouped.append(np.mean(buffer))
        times_selected = np.array(grouped)

    #print("Knacks-Zeiten (s):", times_selected)
    #print("Anzahl der Knacksen:", len(indices))
    return [
        {"name": "Knacks2", "times": times_selected}
    ]

def analyse_uebersteuerung(y, sr, audio_file):

    frame_length = int(0.05 * sr)
    hop_length = int(0.05 * sr)

    max_amp = np.max(np.abs(y))
    if max_amp >= 1.0:
        high_thresh = 1.0
        low_thresh = 0.98
    else:
        high_thresh = 0.95
        low_thresh = 0.93

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    max_abs_per_frame = np.max(np.abs(frames), axis=0)

    ueber = []
    ueber_end = []
    high = False
    for i in range(0, len(max_abs_per_frame)):
        if high_thresh==1.0:
            if any(max_abs_per_frame[i:i+3] >= high_thresh) and all(max_abs_per_frame[i:i+3] >= low_thresh) and high is False:
                ueber.append(i)
                high = True
            if max_abs_per_frame[i] < low_thresh and high is True:
                high = False
                ueber_end.append(i)
        else: 
            if all(max_abs_per_frame[i:i+3] >= high_thresh) and high is False:
                ueber.append(i)
                high = True
            if max_abs_per_frame[i] < low_thresh and high is True:
                high = False
                ueber_end.append(i)

    ueber_times = librosa.frames_to_time(ueber, sr=sr, hop_length=hop_length)
    ueber_end_times = librosa.frames_to_time(ueber_end, sr=sr, hop_length=hop_length)
    # print("Übersteuerungs-Beginn Zeiten (s):", ueber_times)
    # print("Anzahl Übersteuerungen:", len(ueber_end_times))

    return [
        {"name": "Übersteuerung Start", "times": ueber_times},
        {"name": "Übersteuerung Ende", "times": ueber_end_times}
    ]

def analyse_rauschen(y, sr, audio_file):
    events = []

    # Kurzzeitenergie (RMS)
    hop_length = 512
    frame_length = 1024

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Robuste Dynamik (Perzentile statt Min/Max)
    low = np.percentile(rms, 10)
    high = np.percentile(rms, 90)

    # Schutz vor Division durch 0
    if low <= 1e-10:
        return []

    dynamic_range_db = 20 * np.log10(high / low)

    # Entscheidungsregel
    # < 20 dB → Rauschen dominiert
    # >= 20 dB → Sprache dominiert
    
    if dynamic_range_db < 20:
        events.append({"name": "Rauschen", "times": []})

    return events


# -----------------------------------------------------
# Plotfunktion
# -----------------------------------------------------
def plot_overview(y, sr, events, audio_file):

    times = np.arange(len(y)) / sr

    plt.figure(figsize=(14, 5))

    # gesamtes Audio
    plt.plot(times, y, color="lightgray", label="Waveform")

    color_map = {
        "Lautstärke hoch": "red",
        "Lautstärke runter": "green",
        "Knacks": "orange",
        "Knacks2": "yellow",
        "Übersteuerung Start": "purple",
        "Übersteuerung Ende": "blue"
    }

    plotted_labels = set()

    warning_names = ["Kein Ton", "Sehr leise", "Rauschen"]

    # Warnungen sammeln (einmalig)
    warnings = []

    for ev in events:
        name = ev["name"]
        color = color_map.get(name, "black")


        if ev["name"] in warning_names and ev["name"] not in warnings:
            warnings.append(ev["name"])
        
        y_pos = 0.95
        for w in warnings:
            plt.text(
                0.01,
                y_pos,
                w,
                transform=plt.gca().transAxes,
                fontsize=12,
                fontweight="bold",
                color="red"
            )
            y_pos -= 0.06   # Abstand zwischen den Zeile

        for t in ev["times"]:
            if name not in plotted_labels:
                plt.axvline(t, color=color, linestyle="--", alpha=0.8, label=name)
                plotted_labels.add(name)
            else:
                plt.axvline(t, color=color, linestyle="--", alpha=0.8)
                
            plt.text(
            t,
            0.98,
            f"{t:.2f}s",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="right",
            transform=plt.gca().get_xaxis_transform(),
            fontsize=8,
            color=color,
            alpha=0.8
            )
    

    plt.title(audio_file)
    plt.xlabel("Zeit (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(output_folder, f"{audio_file}_overview.png")
    plt.savefig(out)
    plt.close()

    print("➡ Gesamtplot gespeichert:", out)


def make_summary_table(audio_file, events):

    row = {
        "Datei": audio_file,
        "kein Ton": "",
        "Sehr leise": "",
        "Rauschen": "",
        "Übersteuerung": "",
        "Knacks Typ 1": "",
        "Knacks Typ 2": "",
        "Lautstärke": ""
    }

    ueber_start = []
    knacks1 = []
    knacks2 = []
    lauter = []
    leiser = []

    for ev in events:
        name = ev["name"]
        times = ev.get("times", [])

        if name == "Kein Ton":
            row["kein Ton"] = "Kein Ton"

        if name == "Sehr leise":
            row["Sehr leise"] = "Sehr leise"

        if name == "Rauschen":
            row["Rauschen"] = "Rauschen erkannt"

        if name == "Knacks" and len(times) > 0:
            knacks1.extend(times)

        if name == "Knacks2" and len(times) > 0:
            knacks2.extend(times)

        if name == "Übersteuerung Start" and len(times) > 0:
            ueber_start.extend(times)

        if name == "Lautstärke hoch" and len(times) > 0:
            lauter.extend(times)

        if name == "Lautstärke runter" and len(times) > 0:
            leiser.extend(times)

    # -------- Texte bauen --------
    if knacks1:
        row["Knacks Typ 1"] = ", ".join(f"{t:.2f}" for t in knacks1)

    if knacks2:
        row["Knacks Typ 2"] = ", ".join(f"{t:.2f}" for t in knacks2)

    if ueber_start:
        row["Übersteuerung"] = (
            "Startzeiten:\n" +
            ", ".join(f"{t:.2f}" for t in ueber_start)
        )

    if lauter or leiser:
        txt = []
        if lauter:
            txt.append("Lauter:\n" + ", ".join(f"{t:.2f}" for t in lauter))
        if leiser:
            txt.append("Leiser:\n" + ", ".join(f"{t:.2f}" for t in leiser))
        row["Lautstärke"] = "\n".join(txt)

    return row

# -----------------------------------------------------
# Alles
# -----------------------------------------------------
def run_all_analyses(filepath):

    audio_file = os.path.basename(filepath)
    print("\nVerarbeite:", audio_file)

    y, sr = load_audio(filepath)

    skip, events = check_no_signal(y, sr)

    all_events = []
    all_events += events

    if skip:
        plot_overview(y, sr, all_events, audio_file)
        summary_results.append(make_summary_table(audio_file, all_events))

        return

    all_events += analyse_lautstaerke(y, sr, audio_file)
    all_events += analyse_knacks(y, sr, audio_file)
    all_events += analyse_knacks2(y, sr, audio_file)
    all_events += analyse_uebersteuerung(y, sr, audio_file)
    all_events += analyse_rauschen(y, sr, audio_file)

    plot_overview(y, sr, all_events, audio_file)
    summary_results.append(make_summary_table(audio_file, all_events))



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
#         run_all_analyses(os.path.join(input_folder, filename))
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
        run_all_analyses(os.path.join(input_folder, filename))
    except Exception as e:
        print("Fehler bei Datei:", filename)
        print(e)


if summary_results:
    df = pd.DataFrame(summary_results)

    def color_cells(val):
        if val is None or val == "":
            return "background-color: #c6efce"  # grün
        return "background-color: #ffc7ce"      # rot

    styled = df.style.applymap(
        color_cells,
        subset=df.columns[1:]
    )

    out_xlsx = os.path.join(output_folder, "summary_farbig.xlsx")
    styled.to_excel(out_xlsx, engine="openpyxl", index=False)

    print("➡ Excel-Übersicht gespeichert:", out_xlsx)


print("\nFertig!")

