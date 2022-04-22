import numpy as np
import utils.datamaker
import matplotlib.pyplot as plt
import sklearn
import matplotlib.animation as animation


def mean_A_H(i, params, path):
    pre_syn = params[0]
    threshold = params[1]
    chain_len = params[2]
    startE0 = params[3]
    # tLimits = [200, 300]
    tLimits = [100]
    path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_700.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0])
    my_data = np.genfromtxt(path_load1, delimiter=',')
    print("Params: {}".format(params))
    last_few = 5 + chain_len
    repeating = 5

    # Em = my_data[:, -6]
    # ENT = my_data[:, -5]
    # B = my_data[:, -4]
    # NULL = my_data[:, -3]
    TIME = my_data[:, -2]
    # FEA = my_data[:, -1]

    i = 0
    while i < my_data[0, :].shape[0] - last_few:
        # globals()["I{}".format(int(i / repeating))] = my_data[:, i]
        globals()["A{}".format(int(i / repeating))] = my_data[:, i+1]
        globals()["H{}".format(int(i / repeating))] = my_data[:, i+2]
        i += repeating

    means_H = []
    means_A = []
    for s in range(0, pre_syn):
        time_domain = TIME[-1] - TIME[0]

        int_trapz = np.trapz(globals()["H{}".format(s)], TIME)
        mean = int_trapz/time_domain
        print("H{} mean trapz: {}".format(s, mean))
        means_H.append(mean)

        int_trapz = np.trapz(globals()["A{}".format(s)], TIME)
        mean = int_trapz/time_domain
        print("A{} mean trapz: {}".format(s, mean))
        means_A.append(mean)

    np.save(path + "/means_A.npy", means_A, allow_pickle=True)
    np.save(path + "/means_H.npy", means_H, allow_pickle=True)

    return means_H, means_A


def new_mean_stdev(i, params, path):
    pre_syn = params[0]
    threshold = params[1]
    chain_len = params[2]
    startE0 = params[3]
    bolus = params[4]
    # tLimits = [200, 300]
    # tLimits = [200]
    # tLimits = [300]
    tLimits = [300]
    path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_700.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0])
    # path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_1000.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0])
    # path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_1000_bol{}.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0], str(bolus))
    # path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_2000_bol{}.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0], str(bolus))
    # path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_3000_bol{}.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0], str(bolus))
    # path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_1000_steep{}.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0], str(bolus))
    # path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_1500_steep{}.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0], str(bolus))
    my_data = np.genfromtxt(path_load1, delimiter=',')
    print("Params: {}".format(params))

    # if chain_len != 5:
    #     # path1 = "/home/cug/jf330/Paper2/TEST_A_300_cont"
    #     # path2 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_A_300_new"
    #
    #     path1 = "/home/cug/jf330/Paper2/TEST_B_300_cont"
    #     path2 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_B_300_new"
    #
    #     try:
    #         path_load1 = path1 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_100.txt".format(pre_syn, threshold, chain_len, startE0, tLimits[0])
    #         my_data = np.genfromtxt(path_load1, delimiter=',')
    #     except:
    #         path_load1 = path2 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_100.txt".format(pre_syn, threshold, chain_len, startE0, tLimits[0])
    #         my_data = np.genfromtxt(path_load1, delimiter=',')
    # else:
    #     tLimits = [200, 300]
    #     # path1 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST"
    #     # path2 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_A_300_new"
    #
    #     path1 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_B"
    #     path2 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_B_300_new"
    #
    #     try:
    #         path_load1 = path1 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_150.txt".format(pre_syn,
    #                                                                                                       threshold,
    #                                                                                                       chain_len,
    #                                                                                                       startE0,
    #                                                                                                       tLimits[0])
    #         my_data1 = np.genfromtxt(path_load1, delimiter=',')
    #     except:
    #         path_load1 = path1 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_100.txt".format(pre_syn,
    #                                                                                                       threshold,
    #                                                                                                       chain_len,
    #                                                                                                       startE0,
    #                                                                                                       tLimits[0])
    #         my_data1 = np.genfromtxt(path_load1, delimiter=',')
    #
    #     path_load2 = path2 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_200.txt".format(pre_syn,
    #                                                                                                   threshold,
    #                                                                                                   chain_len,
    #                                                                                                   startE0,
    #                                                                                                   tLimits[1])
    #     my_data2 = np.genfromtxt(path_load2, delimiter=',')
    #     # my_data2[:, -2] += 200
    #
    #     my_data = np.concatenate((my_data1, my_data2), axis=0)
    #     del my_data1
    #     del my_data2
    #
    # path_load2 = path2 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_100.txt".format(pre_syn, threshold, chain_len, startE0, tLimits[1])
    # my_data2 = np.genfromtxt(path_load2, delimiter=',')
    # my_data = np.genfromtxt(path_load1, delimiter=',')
    # my_data2[:, -2] += 200

    # my_data = np.concatenate((my_data1, my_data2), axis=0)
    # del my_data1
    # del my_data2
    # print(my_data.shape)

    # all_Es = np.linspace(1, chain_len, chain_len)
    last_few = 5 + chain_len
    repeating = 5

    # Em = my_data[:, -6]
    # ENT = my_data[:, -5]
    # B = my_data[:, -4]
    # NULL = my_data[:, -3]
    TIME = my_data[:, -2]
    # FEA = my_data[:, -1]

    i = 0
    while i < my_data[0, :].shape[0] - last_few:
        # globals()["I{}".format(int(i / repeating))] = my_data[:, i]
        # globals()["A{}".format(int(i / repeating))] = my_data[:, i+1]
        globals()["H{}".format(int(i / repeating))] = my_data[:, i+2]
        i += repeating

    means = []
    stdevs = []
    for s in range(0, pre_syn):
        int_trapz = np.trapz(globals()["H{}".format(s)], TIME)
        time_domain = TIME[-1] - TIME[0]
        # print("Total t: {}".format(time_domain))

        mean = int_trapz/time_domain
        print("H{} mean trapz: {}".format(s, mean))
        means.append(mean)

        # mean_np = np.mean(globals()["H{}".format(s)])
        # print("Mean np: {}".format(mean_np))

        std_np = np.std(globals()["H{}".format(s)])
        print("H{} St. dev. np: {}".format(s, std_np))
        stdevs.append(std_np)

    return means, stdevs


def new_mean_stdev_cell(i, params, path):
    pre_syn = params[0]
    threshold = params[1]
    chain_len = params[2]
    startE0 = params[3]
    bolus = params[4]
    # tLimits = [200, 300]
    tLimits = [100]
    # tLimits = [300]
    # path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_1000_bol{}.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0], str(bolus))
    # path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_2000_bol{}.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0], str(bolus))
    # path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_3000_bol{}.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0], str(bolus))
    # path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_1000_steep{}.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0], str(bolus))
    path_load1 = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_1500_steep{}.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0], str(bolus))

    my_data = np.genfromtxt(path_load1, delimiter=',')
    print("Params: {}".format(params))

    # all_Es = np.linspace(1, chain_len, chain_len)
    last_few = 5 + 2
    repeating = 8

    TIME = my_data[:, -2]
    i = 0
    while i < my_data[0, :].shape[0] - last_few:
        globals()["H{}".format(int(i / repeating))] = my_data[:, i+2]
        i += repeating

    means = []
    stdevs = []
    for s in range(0, pre_syn):
        int_trapz = np.trapz(globals()["H{}".format(s)], TIME)
        time_domain = TIME[-1] - TIME[0]
        # print("Total t: {}".format(time_domain))

        mean = int_trapz/time_domain
        print("H{} mean trapz: {}".format(s, mean))
        means.append(mean)

        # mean_np = np.mean(globals()["H{}".format(s)])
        # print("Mean np: {}".format(mean_np))

        std_np = np.std(globals()["H{}".format(s)])
        print("H{} St. dev. np: {}".format(s, std_np))
        stdevs.append(std_np)

    return means, stdevs


def new_IOD(i, params, path):
    pre_syn = params[0]
    threshold = params[1]
    chain_len = params[2]
    startE0 = params[3]
    tLimits = [700]

    print("Params: {}".format(params))
    # path1 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST"
    path2 = "/home/cug/jf330/Paper2/TEST_IOD_TIME"

    # path1 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_B"
    # path2 = "/home/cug/jf330/Paper2/TEST_B_300"

    # try:
    #     path_load1 = path1 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_150.txt".format(pre_syn, threshold, chain_len, startE0, tLimits[0])
    #     my_data1 = np.genfromtxt(path_load1, delimiter=',')
    # except:
    #     path_load1 = path1 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_100.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0])
    #     my_data1 = np.genfromtxt(path_load1, delimiter=',')

    path_load2 = path2 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.txt".format(pre_syn, threshold, chain_len, startE0, tLimits[0])
    my_data = np.genfromtxt(path_load2, delimiter=',')
    # my_data2[:, -2] += 200

    # my_data = np.concatenate((my_data1, my_data2), axis=0)
    # del my_data1
    # del my_data2

    # all_Es = np.linspace(1, chain_len, chain_len)
    last_few = 5 + chain_len
    repeating = 5

    # Em = my_data[:, -6]
    # ENT = my_data[:, -5]
    # B = my_data[:, -4]
    # NULL = my_data[:, -3]
    TIME = my_data[:, -2]
    # FEA = my_data[:, -1]

    i = 0
    ALL_H = []
    while i < my_data[0, :].shape[0] - last_few:
        # globals()["I{}".format(int(i / repeating))] = my_data[:, i]
        # globals()["A{}".format(int(i / repeating))] = my_data[:, i+1]
        globals()["H{}".format(int(i / repeating))] = my_data[:, i + 2]
        ALL_H.append(my_data[:, i + 2])
        i += repeating

    ALL_H = np.array(ALL_H)
    iod = []
    # tests = np.linspace(1, int(sum(tLimits)), int(sum(tLimits)/10))
    tests = [0, 100, 200, 300, 400, 500, 600]
    for t in tests:
        H_means = []
        for s in range(0, pre_syn):
            idx_s = find_nearest(TIME, t)
            idx_e = find_nearest(TIME, t + 100)
            int_trapz = np.trapz(ALL_H[s, idx_s:idx_e], TIME[idx_s:idx_e])
            time_domain = 100
            # print("Total t: {}".format(time_domain))

            mean = int_trapz / time_domain
            print("H{} mean trapz: {}".format(s, mean))
            H_means.append(mean)

        var = np.var(H_means)
        mean = np.mean(H_means)
        iod.append(var / mean)

    np.save(path2 + "/pap_figs/iod_time_vol{}_ch_{}.npy".format(int(threshold/5), chain_len), iod)
    return iod


def new_MI_EmA(i, params, path):
    pre_syn = params[0]
    threshold = params[1]
    chain_len = params[2]
    startE0 = params[3]
    tLimits = [200, 300]

    print("Params: {}".format(params))
    # path1 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST"
    # path2 = "/home/cug/jf330/Paper2/TEST_A_300"

    path1 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_B"
    path2 = "/home/cug/jf330/Paper2/TEST_B_300"

    try:
        path_load1 = path1 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_150.txt".format(pre_syn, threshold, chain_len, startE0, tLimits[0])
        my_data1 = np.genfromtxt(path_load1, delimiter=',')
    except:
        path_load1 = path1 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_100.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0])
        my_data1 = np.genfromtxt(path_load1, delimiter=',')

    path_load2 = path2 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_200.txt".format(pre_syn, threshold, chain_len, startE0, tLimits[1])
    my_data2 = np.genfromtxt(path_load2, delimiter=',')
    # my_data2[:, -2] += 200

    my_data = np.concatenate((my_data1, my_data2), axis=0)
    del my_data1
    del my_data2

    # all_Es = np.linspace(1, chain_len, chain_len)
    last_few = 5 + chain_len
    repeating = 5

    Em = my_data[:, -6]
    # ENT = my_data[:, -5]
    # B = my_data[:, -4]
    # NULL = my_data[:, -3]
    # TIME = my_data[:, -2]
    # FEA = my_data[:, -1]

    i = 0
    while i < my_data[0, :].shape[0] - last_few:
        # globals()["I{}".format(int(i / repeating))] = my_data[:, i]
        globals()["A{}".format(int(i / repeating))] = my_data[:, i+1]
        # globals()["H{}".format(int(i / repeating))] = my_data[:, i+2]
        i += repeating

    all_mi = []
    for s in range(0, pre_syn):
        mi = sklearn.metrics.mutual_info_score(globals()["A{}".format(s)], Em)
        all_mi.append(mi)
        print("A{}-Em MI: {}".format(s, mi))

    return all_mi


def new_entropy(i, params, path):
    pre_syn = params[0]
    threshold = params[1]
    chain_len = params[2]
    startE0 = params[3]
    tLimits = [200, 300]

    print("Params: {}".format(params))
    # path1 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST"
    # path2 = "/home/cug/jf330/Paper2/TEST_A_300"

    path1 = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_B"
    path2 = "/home/cug/jf330/Paper2/TEST_B_300"

    try:
        path_load1 = path1 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_150.txt".format(pre_syn, threshold, chain_len, startE0, tLimits[0])
        my_data1 = np.genfromtxt(path_load1, delimiter=',')
    except:
        path_load1 = path1 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_100.txt".format(pre_syn, threshold,chain_len, startE0, tLimits[0])
        my_data1 = np.genfromtxt(path_load1, delimiter=',')

    path_load2 = path2 + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_200.txt".format(pre_syn, threshold, chain_len, startE0, tLimits[1])
    my_data2 = np.genfromtxt(path_load2, delimiter=',')
    my_data2[:, -2] += 200

    my_data = np.concatenate((my_data1, my_data2), axis=0)
    del my_data1
    del my_data2

    all_Es = np.linspace(1, chain_len, chain_len)
    last_few = 5 + chain_len
    repeating = 5

    # Em = my_data[:, -6]
    ENT = my_data[:, -5]
    # B = my_data[:, -4]
    # NULL = my_data[:, -3]
    TIME = my_data[:, -2]
    # FEA = my_data[:, -1]

    m, b = np.polyfit(TIME, ENT, 1)

    return m


def plot_example_assoc(i, params, path):
    pre_syn = params[0]
    threshold = params[1]
    chain_len = params[2]
    startE0 = params[3]
    tLimits = [1]

    print("Params: {}".format(params))

    path_load = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.txt".format(pre_syn, threshold, chain_len, startE0, tLimits[0])
    my_data = np.genfromtxt(path_load, delimiter=',')

    all_Es = np.linspace(1, chain_len, chain_len)
    last_few = 5 + chain_len
    repeating = 5

    Em = my_data[:, -6]
    # ENT = my_data[:, -5]
    B = my_data[:, -4]
    # NULL = my_data[:, -3]
    TIME = my_data[:, -2]
    # FEA = my_data[:, -1]

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=.08, bottom=.17, right=.97, top=.95)
    ax1 = fig.add_subplot(5, 1, 1)
    ax1b = fig.add_subplot(5, 1, 2, sharex=ax1)
    ax2 = fig.add_subplot(5, 1, 3, sharex=ax1)
    ax3 = fig.add_subplot(5, 1, 4, sharex=ax1)
    ax4 = fig.add_subplot(5, 1, 5, sharex=ax1)

    ax1.set_ylabel("A1")
    ax1b.set_ylabel("A2")
    ax2.set_ylabel("B")
    ax2.axhline(y=threshold, linestyle="--", color="k", zorder=1)
    # ax3.set_ylabel("Em")
    ax3.set_ylabel("$\mathcal{E}$")
    ax3.yaxis.set_label_coords(-0.07, 0.6)
    ax4.set_ylabel("H")
    ax4.set_xlabel("time")

    i = 0
    idx = 0
    colors = ["red", "blue"]
    while i < my_data[0, :].shape[0] - last_few:
        # globals()["I{}".format(int(i / repeating))] = my_data[:, i]
        # globals()["A{}".format(int(i / repeating))] = my_data[:, i+1]
        # globals()["H{}".format(int(i / repeating))] = my_data[:, i+2]
        ax4.plot(TIME, my_data[:, i+2], color=colors[idx], label="Weight {}".format(idx + 1))
        if idx == 0:
            ax1.plot(TIME, my_data[:, i + 1], color=colors[idx], label="Input {}".format(idx + 1))
        else:
            ax1b.plot(TIME, my_data[:, i + 1], color=colors[idx], label="Input {}".format(idx + 1))

        i += repeating
        idx += 1

    # ax3.plot(TIME, Em, color="black")
    # ax3.scatter(TIME, Em, color="black", marker="x", s=1)
    ax3.plot(TIME, Em, color="black", label="Activation")
    ax3.set_yticks(np.linspace(0, startE0, startE0+1, dtype=int))

    ax2.plot(TIME, B, color="black", label="Membrane \n potential")
    # ax2.plot(TIME, B, color="black", label="State")
    # ax2.scatter(TIME, B, color="black", marker="x", s=1, label="Membrane \n potential")

    # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax1b.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax4.legend(loc='lower right', prop={'size': 6})

    plt.show()

def plot_example_assoc_cell(i, params, path):
    pre_syn = params[0]
    threshold = params[1]
    chain_len = params[2]
    startE0 = params[3]
    steep = params[4]
    tLimits = [1]

    print("Params: {}".format(params))

    path_load = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_steep{}.txt".format(pre_syn, threshold, chain_len,
                                                                                   startE0, tLimits[0], steep)
    my_data = np.genfromtxt(path_load, delimiter=',')

    last_few = 5 + 2
    repeating = 8

    Em = my_data[:, -3]
    # ENT = my_data[:, -5]
    B = my_data[:, -6]
    # NULL = my_data[:, -3]
    TIME = my_data[:, -2]
    # FEA = my_data[:, -1]

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=.08, bottom=.17, right=.97, top=.95)
    ax1 = fig.add_subplot(5, 1, 1)
    ax1b = fig.add_subplot(5, 1, 2, sharex=ax1)
    ax2 = fig.add_subplot(5, 1, 3, sharex=ax1)
    ax3 = fig.add_subplot(5, 1, 4, sharex=ax1)
    ax4 = fig.add_subplot(5, 1, 5, sharex=ax1)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1b.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    # plt.setp(ax1.get_xticklabels(), visible=False)

    ax1.set_ylabel("A1")
    ax1b.set_ylabel("A2")
    ax2.set_ylabel("B")
    ax2.axhline(y=threshold, linestyle="--", color="k", zorder=1)
    # ax3.set_ylabel("Em")
    ax3.set_ylabel("$\mathcal{E}$")
    # ax3.yaxis.set_label_coords(-0.07, 0.6)
    ax4.set_ylabel("H")
    ax4.set_xlabel("time")

    i = 0
    idx = 0
    colors = ["red", "blue"]
    while i < my_data[0, :].shape[0] - last_few:
        # globals()["I{}".format(int(i / repeating))] = my_data[:, i]
        # globals()["A{}".format(int(i / repeating))] = my_data[:, i+1]
        # globals()["H{}".format(int(i / repeating))] = my_data[:, i+2]
        ax4.plot(TIME, my_data[:, i + 2], color=colors[idx], label="Weight {}".format(idx + 1))
        if idx == 0:
            ax1.plot(TIME, my_data[:, i + 1], color=colors[idx], label="Input {}".format(idx + 1))
        else:
            ax1b.plot(TIME, my_data[:, i + 1], color=colors[idx], label="Input {}".format(idx + 1))

        i += repeating
        idx += 1

    # ax3.plot(TIME, Em, color="black")
    # ax3.scatter(TIME, Em, color="black", marker="x", s=1)
    ax3.plot(TIME, Em, color="black", label="Activation")
    # ax3.set_yticks(np.linspace(0, startE0, startE0 + 1, dtype=int))

    ax2.plot(TIME, B, color="black", label="Membrane \n potential")
    # ax2.plot(TIME, B, color="black", label="State")
    # ax2.scatter(TIME, B, color="black", marker="x", s=1, label="Membrane \n potential")

    # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax1b.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax4.legend(loc='lower right', prop={'size': 6})

    plt.show()


def plot_example_learn(i, params, path):
    pre_syn = params[0]
    threshold = params[1]
    chain_len = params[2]
    startE0 = params[3]
    tLimits = [0]

    print("Params: {}".format(params))

    path_load = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.txt".format(pre_syn, threshold, chain_len, startE0, tLimits[0])
    my_data = np.genfromtxt(path_load, delimiter=',')

    all_Es = np.linspace(1, chain_len, chain_len)
    last_few = 5 + chain_len
    repeating = 5

    Em = my_data[:, -6]
    # ENT = my_data[:, -5]
    B = my_data[:, -4]
    # NULL = my_data[:, -3]
    TIME = my_data[:, -2]
    # FEA = my_data[:, -1]

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=.06, bottom=.18, right=.98, top=.95, wspace=0.25)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, sharex=ax1)
    ax3 = fig.add_subplot(1, 3, 3, sharex=ax1)

    ax1.set_ylabel("B")
    ax1.axhline(y=threshold, linestyle="--", color="k", zorder=1)
    ax3.set_ylabel("H")
    ax2.set_ylabel("$\mathcal{E}$")
    # ax3.yaxis.set_label_coords(-0.07,0.6)
    ax1.set_xlabel("time")
    ax2.set_xlabel("time")
    ax3.set_xlabel("time")


    i = 0
    idx = 0
    # colors = ["red", "blue"]
    # while i < my_data[0, :].shape[0] - last_few:
    #     globals()["I{}".format(int(i / repeating))] = my_data[:, i]
    #     globals()["A{}".format(int(i / repeating))] = my_data[:, i+1]
    #     globals()["H{}".format(int(i / repeating))] = my_data[:, i+2]
    #     ax1.plot(TIME, my_data[:, i+1], color=colors[idx], label="Input {}".format(idx + 1))
        # i += repeating
        # idx += 1
    ax3.plot(TIME, my_data[:, 2], color="black")

    # ax3.plot(TIME, Em, color="black")
    # ax3.scatter(TIME, Em, color="black", marker="x", s=1)
    ax2.plot(TIME, Em,  "--x", color="black", markersize=1)
    # ax3.set_yticks(np.linspace(0, startE0, startE0+1, dtype=int))

    # ax2.plot(TIME, B, color="black")
    ax1.scatter(TIME, B, color="black", marker="x", s=1)

    plt.show()


def animate_plot():
    path = "/Users/jf330/CELL_exp/123_test"
    params = [5, 40, 4, 16, path]

    # ani = animation.FuncAnimation(plot_live_CELL, fargs=(params,), interval=100)
    plot_live_CELL2(0, params)

def plot_live_CELL2(i, params):
    pre_syn = params[0]
    threshold = params[1]
    chain_len = params[2]
    startE0 = params[3]
    path = params[4]
    tLimit = 100

    # print("Params: {}".format(params))

    path_load = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.txt".format(pre_syn, threshold, chain_len, startE0, tLimit)
    my_data = np.genfromtxt(path_load, delimiter=',')

    all_Es = np.linspace(1, chain_len, chain_len)
    last_few = 5 + chain_len
    repeating = 7

    Em = my_data[:, -6]
    # ENT = my_data[:, -5]
    B = my_data[:, -4]
    # NULL = my_data[:, -3]
    TIME = my_data[:, -2]
    # FEA = my_data[:, -1]

    fig = plt.figure(figsize=(9, 11))
    fig.subplots_adjust(left=.06, bottom=.18, right=.98, top=.95, wspace=0.25)
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(5, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(5, 1, 4, sharex=ax1)
    ax5 = fig.add_subplot(5, 1, 5, sharex=ax1)

    ax1.set_ylabel("An")

    ax2.set_ylabel("B")
    ax2.axhline(y=threshold, linestyle="--", color="k", zorder=1)

    ax3.set_ylabel("$\mathcal{E}$")

    ax4.set_ylabel("Hn")

    ax5.set_ylabel("An*")

    ax1.set_xlabel("time")
    ax2.set_xlabel("time")
    ax3.set_xlabel("time")
    ax4.set_xlabel("time")
    ax5.set_xlabel("time")

    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0.1, 1, pre_syn+1)]

    i = 0
    idx = 0
    while i < my_data[0, :].shape[0] - last_few:
        # globals()["I{}".format(int(i / repeating))] = my_data[:, i]
        globals()["A{}".format(int(i / repeating))] = my_data[:, i+1]
        globals()["H{}".format(int(i / repeating))] = my_data[:, i+2]
        globals()["A*{}".format(int(i / repeating))] = my_data[:, i+3]

        ax1.plot(TIME, globals()["A{}".format(int(i / repeating))], color=colors[idx], label="Input {}".format(idx + 1))
        ax4.plot(TIME, globals()["H{}".format(int(i / repeating))], color=colors[idx], label="Long term weight {}".format(idx + 1))
        ax5.plot(TIME, globals()["A*{}".format(int(i / repeating))], color=colors[idx], label="Short term weight {}".format(idx + 1))

        i += repeating
        idx += 1

    ax2.plot(TIME, B, color="black")
    ax3.plot(TIME, Em,  "--x", color="black", markersize=1)

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    ax4.legend(loc='best')
    ax5.legend(loc='best')
    plt.show()


def plot_live_CELL(i, params):
    pre_syn = params[0]
    threshold = params[1]
    chain_len = params[2]
    startE0 = params[3]
    path = params[4]
    tLimit = 100

    # print("Params: {}".format(params))

    path_load = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.txt".format(pre_syn, threshold, chain_len, startE0, tLimit)
    my_data = np.genfromtxt(path_load, delimiter=',')

    all_Es = np.linspace(1, chain_len, chain_len)
    last_few = 5 + chain_len
    repeating = 7

    Em = my_data[:, -6]
    # ENT = my_data[:, -5]
    B = my_data[:, -4]
    # NULL = my_data[:, -3]
    TIME = my_data[:, -2]
    # FEA = my_data[:, -1]

    fig = plt.figure(figsize=(9, 11))
    fig.subplots_adjust(left=.06, bottom=.18, right=.98, top=.95, wspace=0.25)
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(5, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(5, 1, 4, sharex=ax1)
    ax5 = fig.add_subplot(5, 1, 5, sharex=ax1)

    ax1.set_ylabel("An")

    ax2.set_ylabel("B")
    ax2.axhline(y=threshold, linestyle="--", color="k", zorder=1)

    ax3.set_ylabel("$\mathcal{E}$")

    ax4.set_ylabel("Hn")

    ax5.set_ylabel("An*")

    ax1.set_xlabel("time")
    ax2.set_xlabel("time")
    ax3.set_xlabel("time")
    ax4.set_xlabel("time")
    ax5.set_xlabel("time")

    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0.1, 1, pre_syn+1)]

    i = 0
    idx = 0
    while i < my_data[0, :].shape[0] - last_few:
        # globals()["I{}".format(int(i / repeating))] = my_data[:, i]
        globals()["A{}".format(int(i / repeating))] = my_data[:, i+1]
        globals()["H{}".format(int(i / repeating))] = my_data[:, i+2]
        globals()["A*{}".format(int(i / repeating))] = my_data[:, i+3]

        ax1.plot(TIME, globals()["A{}".format(int(i / repeating))], color=colors[idx], label="Input {}".format(idx + 1))
        ax4.plot(TIME, globals()["H{}".format(int(i / repeating))], color=colors[idx], label="Long term weight {}".format(idx + 1))
        ax5.plot(TIME, globals()["A*{}".format(int(i / repeating))], color=colors[idx], label="Short term weight {}".format(idx + 1))

        i += repeating
        idx += 1

    ax2.plot(TIME, B, color="black")
    ax3.plot(TIME, Em,  "--x", color="black", markersize=1)

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    ax4.legend(loc='best')
    ax5.legend(loc='best')
    plt.show()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx