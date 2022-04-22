import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
import copy
from itertools import permutations
import time
from numpy.random import choice

def permute(events, to_change):

    for p in range(0, to_change):
        perm_pair = random.sample(range(len(events)), 2)

        ev1_species_temp = events[perm_pair[0]][0]
        ev1_time_temp = events[perm_pair[0]][1]

        ev2_species_temp = events[perm_pair[1]][0]
        ev2_time_temp = events[perm_pair[1]][1]

        events[perm_pair[0]] = (ev1_species_temp, ev2_time_temp)
        events[perm_pair[1]] = (ev2_species_temp, ev1_time_temp)

    events.sort(key=lambda tup: tup[1])
    return events


def poisson_features(species, start_time, stop_time, freq=5, fea_no=1):
    features = []

    for f in range(0, fea_no):
        fea = poisson_input(species, start_time, stop_time, freq)
        features.append(fea)

    return features


def fixed_features(species, start_time, stop_time, no_spikes=1, fea_no=1):
    features = []

    for f in range(0, fea_no):
        events = []
        i = 0
        for spec in species:
            ### Fully random
            t = np.random.uniform(low=start_time, high=stop_time, size=(no_spikes,))[0]

            ### Random in non-overlapping intervals
            # start = (stop_time*i)/len(species)
            # end = (stop_time*(i+1))/len(species)
            # t = np.random.uniform(low=start, high=end, size=(no_spikes,))[0]

            events.append((spec, t))
            i += 1

        events.sort(key=lambda tup: tup[1])
        features.append(events)

    return features


def poisson_input(species, start_time, stop_time, freq=5):

    events = []
    for spec in species:
        t = start_time
        while True:
            t = t+random.expovariate(freq)
            if t > stop_time:
                break
            else:
                events.append((spec, t))

    events.sort(key=lambda tup: tup[1])
    return events


def poisson_input_live(file_path, species, start_time, freq=5):
    t = start_time
    while True:
        t = t+random.expovariate(freq * len(species))
        s = np.choice(species)

        f = open(file_path, "a+")
        f.write(s, t)


def poisson_one_input_live(species, start_time, freq=5):
    # p = [2,2,1,1,1]
    # p = [1,1,1,1,1]
    p = [1,1,1]
    p = np.array(p)
    p = p/sum(p)
    t = start_time
    t = t+random.expovariate(freq * len(species))

    s = np.random.choice(species, p=p)

    return [s, t, 0]


# def poisson_one_input_live_new(species, start_time, freqs):
#     s = np.random.choice(species)
#     s = choice(species, p=weights)
#
#     t = start_time
#     t = t+random.expovariate(freqs[s] * len(species))
#
#     return [s, t, 0]


def poisson_one_input_live_biased(species, start_time, frequencies=[]):
    s = np.random.choice(species)
    freq = frequencies[s]

    t = start_time
    t = t+random.expovariate(freq * len(species))

    return [s, t, 0]


def keyboard_input_live(file_path):
    start_time = time.time()
    import keyboard
    print("Recording...")
    while True:
        if keyboard.read_key() == '0':
            now = time.time() - start_time
            f = open(file_path, "a+")
            f.write("I0, " + str(now) + "\n")

        if keyboard.read_key() == '1':
            now = time.time() - start_time
            f = open(file_path, "a+")
            f.write("I1, " + str(now) + "\n")


def prepare_input(I_species, features, desired_spikes, noise_stdev, stop_time, mean_occur, fea_len):

    all_events = []
    if mean_occur > 0:
        n_fea_occur = np.random.poisson(mean_occur, len(features))
    else:
        n_fea_occur = np.ones(len(features), dtype=int)

    all_fea = sum(n_fea_occur)

    insert_times = time_occurrence(n_fea_occur, stop_time)
    fea_order, resp_order = gen_order_occur(n_fea_occur, desired_spikes)

    start_bg = 0
    for f_idx in range(0, len(fea_order)):

        insert_times[f_idx] = insert_times[f_idx] + (f_idx * fea_len)

        fea = features[fea_order[f_idx]]
        bg = poisson_input(species=I_species, start_time=start_bg, stop_time=insert_times[f_idx], freq=5)

        fea_events = copy.copy(fea)
        for e_idx in range(0, len(fea_events)):
            timing = fea_events[e_idx][1] + insert_times[f_idx] + np.random.normal(0, noise_stdev)
            timing = np.clip(timing, a_min=0, a_max=None)
            fea_events[e_idx] = (fea_events[e_idx][0], timing)

        all_events += bg
        all_events += fea_events

        start_bg = insert_times[f_idx]+fea_len

    bg_end = poisson_input(species=I_species, start_time=start_bg, stop_time=stop_time+(all_fea*fea_len), freq=5)
    all_events += bg_end

    all_events.sort(key=lambda tup: tup[1])

    return all_events, insert_times, fea_order, resp_order


def time_occurrence(n_fea_occur, stop_time):
        return np.sort(np.random.random(np.sum(n_fea_occur)) * stop_time)


def gen_order_occur(n_fea_occur, desired_spikes):
    count = 0
    order = []
    while count < len(n_fea_occur):
        order += [count] * n_fea_occur[count]
        count += 1

    order_fea = np.random.permutation(order)

    order_resp = []
    for i in order_fea:
        order_resp.append(desired_spikes[i])

    return order_fea, order_resp


def calc_error(insert_times, resp_order, fea_order, out_hist, time_hist, fea_len, fea_no):
    error = np.zeros((len(time_hist)))
    spike_est = np.zeros(fea_no)

    start_bg = 0
    for f_idx in range(0, len(insert_times)):
        ### Noisy background activity
        bg_start_idx = find_nearest(time_hist, start_bg)
        bg_end_idx = find_nearest(time_hist, insert_times[f_idx])
        bg_spike_est = np.trapz(out_hist[bg_start_idx:bg_end_idx], time_hist[bg_start_idx:bg_end_idx])
        error[bg_start_idx:bg_end_idx] = bg_spike_est - 0

        ### Feature response
        start_idx = find_nearest(time_hist, insert_times[f_idx])
        end_idx = find_nearest(time_hist, insert_times[f_idx] + fea_len)
        fea_spike_est = np.trapz(out_hist[start_idx:end_idx], time_hist[start_idx:end_idx])
        spike_est[fea_order[f_idx]] += fea_spike_est

        error[start_idx:end_idx] = fea_spike_est - resp_order[f_idx]

        start_bg = insert_times[f_idx] + fea_len

    bg_start_idx = find_nearest(time_hist, start_bg)
    bg_spike_est = np.trapz(out_hist[bg_start_idx:-1], time_hist[bg_start_idx:-1])
    error[bg_start_idx:-1] = bg_spike_est - 0

    return error.tolist(), spike_est


def calc_error_multi(insert_times, resp_order, fea_order, out_hist, time_hist, fea_len, threshold, fea_no):
    error = np.zeros((len(time_hist)))
    spike_est = np.zeros(fea_no)

    start_bg = 0
    for f_idx in range(0, len(insert_times)):
        ### Noisy background activity
        bg_start_idx = find_nearest(time_hist, start_bg)
        bg_end_idx = find_nearest(time_hist, insert_times[f_idx])

        if len(out_hist[bg_start_idx:bg_end_idx]) != 0:
            bg_max = max(out_hist[bg_start_idx:bg_end_idx])
            if bg_max >= threshold:
                error[bg_start_idx:bg_end_idx] = bg_max

        ### Feature response
        start_idx = find_nearest(time_hist, insert_times[f_idx])
        end_idx = find_nearest(time_hist, insert_times[f_idx] + fea_len)

        if len(out_hist[start_idx:end_idx]) == 0:
            fea_max = out_hist[start_idx]
        else:
            fea_max = max(out_hist[start_idx:end_idx])

        spike_est[fea_order[f_idx]] += fea_max

        if fea_max >= (resp_order[f_idx]*threshold) and fea_max < (resp_order[f_idx]*threshold)+(resp_order[f_idx]*threshold):
            error[start_idx:end_idx] = 0
        else:
            error[start_idx:end_idx] = fea_max - (resp_order[f_idx] * threshold)

        start_bg = insert_times[f_idx] + fea_len

    bg_start_idx = find_nearest(time_hist, start_bg)
    if len(out_hist[bg_start_idx:-1]) != 0:
        bg_max = max(out_hist[bg_start_idx:-1])
        if bg_max >= threshold:
            error[bg_start_idx:-1] = 1

    return error.tolist(), spike_est


def calc_error_R_max(insert_times, resp_order, fea_order, out_hist, time_hist, fea_len, threshold, fea_no):
    error = np.zeros((len(time_hist)))
    spike_est = np.zeros(fea_no)

    start_bg = 0
    for f_idx in range(0, len(insert_times)):
        ### Noisy background activity
        bg_start_idx = find_nearest(time_hist, start_bg)
        bg_end_idx = find_nearest(time_hist, insert_times[f_idx])

        if len(out_hist[bg_start_idx:bg_end_idx]) != 0:
            bg_max = max(out_hist[bg_start_idx:bg_end_idx])
            if bg_max >= 0:
                error[bg_start_idx:bg_end_idx] = 1

        ### Feature response
        start_idx = find_nearest(time_hist, insert_times[f_idx])
        end_idx = find_nearest(time_hist, insert_times[f_idx] + fea_len)

        if len(out_hist[start_idx:end_idx]) == 0:
            fea_max = out_hist[start_idx]
        else:
            fea_max = max(out_hist[start_idx:end_idx])

        spike_est[fea_order[f_idx]] += fea_max

        if fea_max > 0 and resp_order[f_idx] > 0:
            error[start_idx:end_idx] = 0
        elif fea_max == 0 and resp_order[f_idx] > 0:
            error[start_idx:end_idx] = -1
        else:
            error[start_idx:end_idx] = 1

        start_bg = insert_times[f_idx] + fea_len

    bg_start_idx = find_nearest(time_hist, start_bg)
    if len(out_hist[bg_start_idx:-1]) != 0:
        bg_max = max(out_hist[bg_start_idx:-1])
        if bg_max >= 0:
            error[bg_start_idx:-1] = 1

    return error.tolist(), spike_est


def calc_error_R_int(insert_times, resp_order, fea_order, out_hist, time_hist, fea_len, fea_no):
    error = np.zeros((len(time_hist)))
    spike_est = np.zeros(fea_no)

    start_bg = 0
    for f_idx in range(0, len(insert_times)):
        ### Noisy background activity
        bg_start_idx = find_nearest(time_hist, start_bg)
        bg_end_idx = find_nearest(time_hist, insert_times[f_idx])
        bg_spike_est = np.trapz(out_hist[bg_start_idx:bg_end_idx], time_hist[bg_start_idx:bg_end_idx])
        error[bg_start_idx:bg_end_idx] = bg_spike_est - 0

        ### Feature response
        start_idx = find_nearest(time_hist, insert_times[f_idx])
        end_idx = find_nearest(time_hist, insert_times[f_idx] + fea_len)
        fea_spike_est = np.trapz(out_hist[start_idx:end_idx], time_hist[start_idx:end_idx])
        spike_est[fea_order[f_idx]] += fea_spike_est

        if resp_order[f_idx] == 0:
            error[start_idx:end_idx] = fea_spike_est - 0
        elif resp_order[f_idx] == 1 and fea_spike_est == 0:
            error[start_idx:end_idx] = -1
        elif resp_order[f_idx] == 1 and fea_spike_est == 1:
            error[start_idx:end_idx] = 0

        start_bg = insert_times[f_idx] + fea_len

    bg_start_idx = find_nearest(time_hist, start_bg)
    bg_spike_est = np.trapz(out_hist[bg_start_idx:-1], time_hist[bg_start_idx:-1])
    error[bg_start_idx:-1] = bg_spike_est - 0

    return error.tolist(), spike_est


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_raster(events, insert_times, fea_order, colors, fea_len):
    for i in range(0, len(insert_times)):
        plt.axvspan(insert_times[i], insert_times[i] + fea_len, alpha=0.2, color=colors[fea_order[i]])

    spec = []
    time = []
    for ev in events:
        spec.append(ev[0])
        time.append(ev[1])
    plt.scatter(time, spec, marker="x")
    plt.show()
