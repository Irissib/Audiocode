
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
def k_means(active_rms):
    active_rms = active_rms.reshape(-1, 1)  # Fix für KMeans
    kmeans = KMeans(n_clusters=3, random_state=0).fit(active_rms)
    centers = np.sort(kmeans.cluster_centers_.flatten())
    thresholds = [(centers[0] + centers[1])/2, (centers[1] + centers[2])/2]
    return thresholds 
#--------------------------
#Analysefunktion-----------------------
#----------------------------------
def analyse_lautstaerke(y, sr, audio_file):

    # Frame-Parameter
    frame_length = int(0.1 * sr)
    hop_length = int(0.05 * sr)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_smooth = scipy.signal.medfilt(rms, kernel_size=7)

    noise_sec=0.3
    noise_frames = int(noise_sec * sr / hop_length)

    noise_th=2*np.mean(rms_smooth[:noise_frames])  # Rauschreferenz

    rms_no_noise = rms_smooth.copy()
    rms_no_noise[rms_no_noise < noise_th] = 0

    active = rms_no_noise[rms_no_noise > 0]

    
    low_region = active[active <= 0.2*np.max(rms_smooth)]

    th_verylow=otsu_threshold(low_region)
    th_verylow = max(th_verylow, noise_th * 1.1)

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
    state = 0  # 0=Stille, 1=leise/mittel, 2=laut
    old_state = 0
    high_idx = []
    low_idx =  []
#
    for idx in range(margin_frames, len(rms_smooth) - margin_frames) :
#
        vol = rms_smooth[idx]
#
        # -------------------------
        # Noise-Check in margin
        # -------------------------
        if vol < noise_th:
            old_state=state
            state = 0
            continue 
#
        local = rms_smooth[idx - margin_frames: idx + margin_frames + 1]

        if state == 1 and old_state==2:
            # Prüfen:größer noise_th
            if np.any(local < noise_th):
                if vol >= th_low:
                    old_state=state
                    state = 2
                elif vol >= th_verylow:
                    state = 1
                    old_state=state
                else:
                    old_state=state
                    state = 0
                continue
#

        else:  
            if np.any(local < th_verylow):
                if vol >= th_low:
                    old_state=state
                    state = 2
                elif vol >= th_verylow:
                    old_state=state
                    state = 1
                else:
                    old_state=state
                    state = 0
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
            if vol >= th_low:
                old_state=state
                state = 2
            elif vol >= th_verylow:
                old_state=state
                state = 1
            else:
                old_state=state
                state = 0
            continue  
#
        diff = abs(np.mean(after) - np.mean(before))
        if diff < 0.5 * mean_amp:
            if vol >= th_low:
                old_state=state
                state = 2
            elif vol >= th_verylow:
                old_state=state
                state = 1
            else:
                old_state=state
                state = 0
            continue 
#
        # -------------------------
        # Zustandslogik (2 Schwellen)
        # -------------------------
        if vol >= th_low:
            new_state = 2
        elif vol >= th_verylow:
            new_state = 1
        else:
            new_state = 0
#
        if new_state > state:
            high_idx.append(idx)
        elif new_state < state:
            low_idx.append(idx) 

        if new_state!=state:
            old_state=state
            state = new_state 
#
    high_times = librosa.frames_to_time(
        high_idx, sr=sr, hop_length=hop_length)
    low_times = librosa.frames_to_time(
        low_idx, sr=sr, hop_length=hop_length)
    
#
    # -------------------------
    # Plot
    # -------------------------
    times = librosa.frames_to_time(
        np.arange(len(rms_smooth)), sr=sr, hop_length=hop_length)
    
#
    plt.figure(figsize=(12, 4))
    plt.plot(times, rms_smooth, label="RMS") 
#
    plt.axhline(noise_th, color='gray', linestyle=':', label='Noise')
    plt.axhline(th_verylow, color='purple', linestyle='--', label='Übergang 1')
    plt.axhline(th_low, color='blue', linestyle='--', label='Übergang 2')
#
    for t in high_times:
        plt.axvline(t, color='red', linestyle='--')
    for t in low_times:
        plt.axvline(t, color='green', linestyle='--') 
#
    plt.legend()
    plt.title("Lautstärkeänderungen (RMS + 2 Übergangsschwellen)")
    plt.tight_layout ()
#
    out = os.path.join(output_folder, f"{audio_file}_lautstaerke3.png")
    plt.savefig(out)
    plt.close ()
#
    print("➡ Gespeichert:", out) 
#
    return [
        {"name": "Lautstärke hoch", "times": high_times},
        {"name": "Lautstärke runter", "times": low_times},
    ]