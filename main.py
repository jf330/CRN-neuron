from stoch_sim.nrm import NRM
from stoch_sim.nrm_live import NRM_live

import os
import numpy as np
import time
import matplotlib.pyplot as plt
import utils.datamaker
import pickle
import copy
from heapq import nlargest
import statistics
import matplotlib as mpl
from itertools import permutations
from mpl_toolkits.mplot3d import Axes3D
import sklearn as sk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import loadmat
from scipy.stats import entropy
from statistics import variance
import gzip


def assoc_run(path, pre_syn=2, chain_len=50, start_E0=5, threshold=0, volume=1, pattern_type="temporal"):
    ### Simulation time limit
    vol = volume
    dt = 1000

    time_limit = 100/dt
    # time_limit = 1000/dt
    # time_limit = 300000/dt
    # time_limit = 550000/dt

    chain_len = int(chain_len)
    # start_E0 = start_E0 * vol

    ### Simulation parameters
    spike_threshold_ratio = 1.3
    # spike_threshold_ratio = 1.2
    # spike_threshold_ratio = 0.8

    fea_len = 0.03  # Feature length (s)
    noise_stdev = 0  # Temporal jitter in features
    fea_no = 1  # Number of features

    # cont_last = True
    cont_last = False

    reporting = True
    # reporting = False

    key_input = False

    print("cont_last = {} \n key_input = {} \n reporting = {}".format(cont_last, key_input, reporting))

    # seed = 0
    seed = np.random.randint(0, 100)
    np.random.seed(seed)
    print("Seed: {}".format(seed))

    ### Generate new weights and prepare reactions
    spec_mol = {}
    reactions = []
    I_species = []
    A_species = []
    H_species = []

    # biased_syn = [0, 1]
    # biased_syn = [0, 1, 2]
    ### REACTION NETWORK SETTINGS
    for i in range(0, pre_syn):
        #
        # if pattern_type == "temporal":
        #     if biased_syn.__contains__(i):
        #         # I_species.append("I{}".format(i))
        #         print("No random spiking for I{}".format(i))
        #     else:
        #         I_species.append("I{}".format(i))
        # else:
        #     if biased_syn.__contains__(i):
        #         I_species.append("I{}".format(i))
        #         print("Additional spiking for I{}".format(i))
        I_species.append("I{}".format(i))

        spec_mol["I{}".format(i)] = 0

        reactions.append([[["I{}".format(i), 1]], [["A{}".format(i), 1]], [dt * 10, [len(reactions) + 1]]])
        reactions.append([[["A{}".format(i), 1]], [["I{}".format(i), 1]], [dt * 0.00000001, [len(reactions) - 1]]])

        A_species.append("A{}".format(i))
        spec_mol["A{}".format(i)] = 0

        reactions.append([[["A{}".format(i), 1]], [["B", 1]], [dt * 0.07, [len(reactions) + 1]]])
        reactions.append([[["B", 1]], [["A{}".format(i), 1]], [dt * 0.00000001, [len(reactions) - 1]]])

        H_species.append("H+{}".format(i))

        ##### START WEIGHTS
        # spec_mol["H+{}".format(i)] = 20 * vol
        if i == 0:
            # spec_mol["H+{}".format(i)] = 100
            spec_mol["H+{}".format(i)] = 0
        else:
            # spec_mol["H+{}".format(i)] = 10
            spec_mol["H+{}".format(i)] = 0

        ##### INCREASE IN H+

        ### PROPER MICHAELIS-MENTEN
        reactions.append([[["A{}".format(i), 1], ["E{}".format(chain_len), 1]], [["E*{}".format(i), 1]], [dt * 0.8 * 1/vol, [len(reactions) + 1]]])
        reactions.append([[["E*{}".format(i), 1]], [["A{}".format(i), 1], ["E{}".format(chain_len), 1]], [dt * 0.00000001, [len(reactions) - 1]]])

        reactions.append([[["E*{}".format(i), 1]], [["E{}".format(chain_len), 1], ["H+{}".format(i), 1]], [dt * 100, [len(reactions) + 1]]])
        reactions.append([[["E{}".format(chain_len), 1], ["H+{}".format(i), 1]], [["E*{}".format(i), 1]], [dt * 0.00000001 * 1/vol, [len(reactions) - 1]]])

        spec_mol["E*{}".format(i)] = 0

        ##### DECREASE IN H+
        reactions.append([[["A{}".format(i), 1], ["H+{}".format(i), 1]], [["H*{}".format(i), 1]], [dt * 0.15 * 1/vol, [len(reactions) + 1]]])
        reactions.append([[["H*{}".format(i), 1]], [["A{}".format(i), 1], ["H+{}".format(i), 1]], [dt * 0.00000001, [len(reactions) - 1]]])

        reactions.append([[["H*{}".format(i), 1]], [["H+{}".format(i), 1], ["B", 1]], [dt * 100, [len(reactions) + 1]]])
        reactions.append([[["H+{}".format(i), 1], ["B", 1]], [["H*{}".format(i), 1]], [dt * 0.00000001 * 1/vol, [len(reactions) - 1]]])

        ### A ORIGINAL: Used for most experiments
        reactions.append([[["H+{}".format(i), 1]], [["NULL", 1]], [dt * 0.00002, [len(reactions) + 1]]])
        # reactions.append([[["H+{}".format(i), 1]], [["NULL", 1]], [dt * 0.00003, [len(reactions) + 1]]])
        reactions.append([[["NULL", 1]], [["H+{}".format(i), 1]], [dt * 0.00000001, [len(reactions) - 1]]])

        spec_mol["H*{}".format(i)] = 0

    ### Diffusion of B
    reactions.append([[["B", 1]], [["NULL", 1]], [dt * 0.07, [len(reactions) + 1]]])
    reactions.append([[["NULL", 1]], [["B", 1]], [dt * 0.00000001, [len(reactions) - 1]]])

    ### SEQUENTIAL BINDING
    if threshold == 0:
        k_plus = 1 * 1/vol

        k_minus = 5
        k_minus_last = 0.4

        threshold = int(k_minus / k_plus)
        mol_spike = int(threshold * spike_threshold_ratio)
    else:
        k_plus = 0.1 * 1/vol
        k_minus_last = 0.5
        mol_spike = int(threshold * spike_threshold_ratio)
        k_minus = threshold * k_plus

    if chain_len == 1:
        spec_mol["E{}".format(1)] = start_E0
        ### Forwards
        reactions.append([[["B", 1]], [["E{}".format(1), 1]], [dt * k_plus, [len(reactions) + 1]]])
        ### Backwards
        reactions.append([[["E{}".format(1), 1]], [["B", 1]], [dt * k_minus, [len(reactions) - 1]]])
    else:
        all_Es = np.linspace(1, int(chain_len), int(chain_len), dtype=int)
        for i in all_Es:
            if i == 1:
                spec_mol["E{}".format(i)] = start_E0
            else:
                spec_mol["E{}".format(i)] = 0

        ### ALL BUT LAST Es IN THE CHAIN
        for i in all_Es[:-2]:
            ### Forwards
            reactions.append(
                [[["B", 1], ["E{}".format(i), 1]], [["E{}".format(i + 1), 1]], [dt * k_plus, [len(reactions) + 1]]])
            ### Backwards
            reactions.append(
                [[["E{}".format(i + 1), 1]], [["B", 1], ["E{}".format(i), 1]], [dt * k_minus, [len(reactions) - 1]]])

        ### LAST ONE
        ### Forwards
        reactions.append([[["B", 1], ["E{}".format(all_Es[-2]), 1]], [["E{}".format(chain_len), 1]],
                          [dt * k_plus, [len(reactions) + 1]]])
        ### Backwards
        reactions.append([[["E{}".format(chain_len), 1]], [["B", 1], ["E{}".format(all_Es[-2]), 1]],
                          [dt * k_minus_last, [len(reactions) - 1]]])

    spec_mol["R_fast"] = 0
    spec_mol["B"] = 0
    spec_mol["NULL"] = 1

    ### Load molecular composition from last checkpoint
    if cont_last:
        load_time_limit = 700
        # load_live_file_path = "/home/jf330/Paper2/paper_results_new/28_05_A/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.npy".format(pre_syn, threshold, chain_len,int(start_E0),int(load_time_limit))
        # load_live_file_path = "/home/jf330/Paper2/paper_results_new/28_05_A/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.npy".format(pre_syn,
        #                                                                                                                                            threshold,
        #                                                                                                                                            chain_len,
        #                                                                                                                                            int(start_E0),
        #                                                                                                                                            int(load_time_limit),
        #                                                                                                                                            int(250))

        load_live_file_path = path + "/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.npy".format(pre_syn,
        # load_live_file_path = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_B_300_new/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.npy".format(pre_syn,
        # load_live_file_path = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_A_300_new/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.npy".format(pre_syn,
                                                                                                               threshold,
                                                                                                               chain_len,
                                                                                                               int(start_E0),
                                                                                                               int(load_time_limit))
        #
        # load_live_file_path = path + "/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.npy".format(pre_syn, threshold, chain_len, int(start_E0), int(load_time_limit))
        H_load = np.load(load_live_file_path)
        # try:
        #     load_live_file_path = "/home/cug/jf330/Paper2/paper_results_gilda/TEST/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.npy".format(pre_syn, threshold, chain_len,int(start_E0),int(load_time_limit), 100)
        #     H_load = np.load(load_live_file_path)
        # except:
        #     pass
        #     load_live_file_path = "/home/cug/jf330/Paper2/paper_results_gilda/TEST/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.npy".format(pre_syn, threshold, chain_len, int(start_E0), int(load_time_limit), 150)
        #     H_load = np.load(load_live_file_path)

        for i in range(0, pre_syn):
            spec_mol["H+{}".format(i)] = H_load[i]
            print("H+{} loaded value: {}".format(i, H_load[i]))

        live_file_path = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.txt".format(pre_syn, threshold,
                                                                                                        chain_len,
                                                                                                        int(start_E0),
                                                                                                        int(time_limit),
                                                                                                        int(load_time_limit))
        H_file_path = path + "/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}".format(pre_syn, threshold, chain_len,
                                                                                                int(start_E0), int(time_limit),
                                                                                                        int(load_time_limit))
    else:

        live_file_path = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.txt".format(pre_syn, threshold,
                                                                                           chain_len,
                                                                                           int(start_E0),
                                                                                           int(time_limit))

        H_file_path = path + "/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}".format(pre_syn, threshold, chain_len,
                                                                                      int(start_E0), int(time_limit))

    open(live_file_path, 'w').close()

    ### 12_A
    features = [[('I0', 0.015712458083621678), ('I1', 0.020453385783666335)]]
    print("Feature: {}".format(features))

    ## Simulate CRN LIVE
    nrm = NRM_live_test(pre_syn, spec_mol, reactions, threshold, features[0], dt=dt, mol_spike=mol_spike,
                   live_file_path=live_file_path, reporting=reporting, key_input=key_input, pattern_type=pattern_type)

    nrm.simulate_fixed(H_file_path, I_species, time_limit)


def train_STDP_properHill_nonpatterns_quick(path, pre_syn=3, chain_len=50, start_E0=5, threshold=0, volume=1, pattern_type="temporal", st_ratio=1):
    ### Simulation time limit
    vol = volume
    dt = 1000

    # time_limit = 100000/dt
    time_limit = 300000/dt
    # time_limit = 500000/dt
    # time_limit = 700000/dt
    # time_limit = 1000000/dt

    chain_len = int(chain_len)
    start_E0 = start_E0 * vol

    ### Simulation parameters
    # spike_threshold_ratio = 1.5
    spike_threshold_ratio = 1
    # spike_threshold_ratio = 0.9
    # spike_threshold_ratio = 0.6
    # spike_threshold_ratio = 0.5

    spike_threshold_ratio = st_ratio
    print("STR: {}".format(spike_threshold_ratio))

    fea_len = 0.03  # Feature length (s)
    noise_stdev = 0  # Temporal jitter in features
    fea_no = 1  # Number of features

    cont_last = True
    # cont_last = False

    reporting = True
    # reporting = False

    key_input = False

    print("cont_last = {} \n key_input = {} \n reporting = {}".format(cont_last, key_input, reporting))

    # seed = 0
    seed = np.random.randint(0, 100)
    np.random.seed(seed)
    print("Seed: {}".format(seed))

    ### Generate new weights and prepare reactions
    spec_mol = {}
    reactions = []
    I_species = []
    A_species = []
    H_species = []

    # biased_syn = [0, 1]
    biased_syn = [0]
    # biased_syn = [0, 1, 2, 3, 4]
    ### REACTION NETWORK SETTINGS
    for i in range(0, pre_syn):

        if pattern_type == "temporal":
            if biased_syn.__contains__(i):
                # I_species.append("I{}".format(i))
                print("No random spiking for I{}".format(i))
            else:
                I_species.append("I{}".format(i))
        else:
            # if biased_syn.__contains__(i):
            #     I_species.append("I{}".format(i))
            #     ### 123_A_more
            #     # I_species.append("I{}".format(i))
            #     print("Additional spiking for I{}".format(i))
            I_species.append("I{}".format(i))

        spec_mol["I{}".format(i)] = 0

        reactions.append([[["I{}".format(i), 1]], [["A{}".format(i), 1]], [dt * 10, [len(reactions) + 1]]])
        reactions.append([[["A{}".format(i), 1]], [["I{}".format(i), 1]], [dt * 0.00000001, [len(reactions) - 1]]])

        A_species.append("A{}".format(i))
        spec_mol["A{}".format(i)] = 0

        reactions.append([[["A{}".format(i), 1]], [["B", 1]], [dt * 0.1, [len(reactions) + 1]]])
        reactions.append([[["B", 1]], [["A{}".format(i), 1]], [dt * 0.00000001, [len(reactions) - 1]]])

        H_species.append("H+{}".format(i))

        ##### START WEIGHTS
        # spec_mol["H+{}".format(i)] = 20 * vol
        spec_mol["H+{}".format(i)] = 10 * vol

        ##### INCREASE IN H+

        ### PROPER MICHAELIS-MENTEN
        reactions.append([[["A{}".format(i), 1], ["E{}".format(chain_len), 1]], [["E*{}".format(i), 1]], [dt * 0.05 * 1/vol, [len(reactions) + 1]]])
        reactions.append([[["E*{}".format(i), 1]], [["A{}".format(i), 1], ["E{}".format(chain_len), 1]], [dt * 0.00000001, [len(reactions) - 1]]])

        reactions.append([[["E*{}".format(i), 1]], [["E{}".format(chain_len), 1], ["H+{}".format(i), 1]], [dt * 100, [len(reactions) + 1]]])
        reactions.append([[["E{}".format(chain_len), 1], ["H+{}".format(i), 1]], [["E*{}".format(i), 1]], [dt * 0.00000001 * 1/vol, [len(reactions) - 1]]])

        spec_mol["E*{}".format(i)] = 0

        ##### DECREASE IN H+
        reactions.append([[["A{}".format(i), 1], ["H+{}".format(i), 1]], [["H*{}".format(i), 1]], [dt * 0.001 * 1/vol, [len(reactions) + 1]]])
        reactions.append([[["H*{}".format(i), 1]], [["A{}".format(i), 1], ["H+{}".format(i), 1]], [dt * 0.00000001, [len(reactions) - 1]]])

        reactions.append([[["H*{}".format(i), 1]], [["H+{}".format(i), 1], ["B", 1]], [dt * 100, [len(reactions) + 1]]])
        reactions.append([[["H+{}".format(i), 1], ["B", 1]], [["H*{}".format(i), 1]], [dt * 0.00000001 * 1/vol, [len(reactions) - 1]]])

        ### A ORIGINAL: Used for most experiments
        # reactions.append([[["H+{}".format(i), 1]], [["NULL", 1]], [dt * 0.00002, [len(reactions) + 1]]])
        reactions.append([[["H+{}".format(i), 1]], [["NULL", 1]], [dt * 0.00003, [len(reactions) + 1]]])
        reactions.append([[["NULL", 1]], [["H+{}".format(i), 1]], [dt * 0.00000001, [len(reactions) - 1]]])

        spec_mol["H*{}".format(i)] = 0

    ### Diffusion of B
    reactions.append([[["B", 1]], [["NULL", 1]], [dt * 0.1, [len(reactions) + 1]]])
    reactions.append([[["NULL", 1]], [["B", 1]], [dt * 0.00000001, [len(reactions) - 1]]])

    ### SEQUENTIAL BINDING
    if threshold == 0:
        k_plus = 1 * 1/vol

        k_minus = 5
        k_minus_last = 0.5

        threshold = int(k_minus / k_plus)
        mol_spike = int(threshold * spike_threshold_ratio)
    else:
        k_plus = 0.1 * 1/vol
        k_minus_last = 0.5
        mol_spike = int(threshold * spike_threshold_ratio)
        k_minus = threshold * k_plus

    if chain_len == 1:
        spec_mol["E{}".format(1)] = start_E0
        ### Forwards
        reactions.append([[["B", 1]], [["E{}".format(1), 1]], [dt * k_plus, [len(reactions) + 1]]])
        ### Backwards
        reactions.append([[["E{}".format(1), 1]], [["B", 1]], [dt * k_minus, [len(reactions) - 1]]])
    else:
        all_Es = np.linspace(1, int(chain_len), int(chain_len), dtype=int)
        for i in all_Es:
            if i == 1:
                spec_mol["E{}".format(i)] = start_E0
            else:
                spec_mol["E{}".format(i)] = 0

        ### ALL BUT LAST Es IN THE CHAIN
        for i in all_Es[:-2]:
            ### Forwards
            reactions.append(
                [[["B", 1], ["E{}".format(i), 1]], [["E{}".format(i + 1), 1]], [dt * k_plus, [len(reactions) + 1]]])
            ### Backwards
            reactions.append(
                [[["E{}".format(i + 1), 1]], [["B", 1], ["E{}".format(i), 1]], [dt * k_minus, [len(reactions) - 1]]])

        ### LAST ONE
        ### Forwards
        reactions.append([[["B", 1], ["E{}".format(all_Es[-2]), 1]], [["E{}".format(chain_len), 1]],
                          [dt * k_plus, [len(reactions) + 1]]])
        ### Backwards
        reactions.append([[["E{}".format(chain_len), 1]], [["B", 1], ["E{}".format(all_Es[-2]), 1]],
                          [dt * k_minus_last, [len(reactions) - 1]]])

    spec_mol["R_fast"] = 0
    spec_mol["B"] = 0
    spec_mol["NULL"] = 1

    ### Load molecular composition from last checkpoint
    if cont_last:
        load_time_limit = 700
        # load_time_limit = 1000
        # load_live_file_path = "/home/jf330/Paper2/paper_results_new/28_05_A/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.npy".format(pre_syn, threshold, chain_len,int(start_E0),int(load_time_limit))
        # load_live_file_path = "/home/jf330/Paper2/paper_results_new/28_05_A/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.npy".format(pre_syn,
        #                                                                                                                                            threshold,
        #                                                                                                                                            chain_len,
        #                                                                                                                                            int(start_E0),
        #                                                                                                                                            int(load_time_limit),
        #                                                                                                                                            int(250))

        load_live_file_path = path + "/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.npy".format(pre_syn,
        # load_live_file_path = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_B_300_new/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.npy".format(pre_syn,
        # load_live_file_path = "/home/cug/jf330/Paper2/paper_results_gilda/TEST_A_300_new/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.npy".format(pre_syn,
                                                                                                               threshold,
                                                                                                               chain_len,
                                                                                                               int(start_E0),
                                                                                                               int(load_time_limit))
        #
        # load_live_file_path = path + "/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.npy".format(pre_syn, threshold, chain_len, int(start_E0), int(load_time_limit))
        H_load = np.load(load_live_file_path)
        # try:
        #     load_live_file_path = "/home/cug/jf330/Paper2/paper_results_gilda/TEST/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.npy".format(pre_syn, threshold, chain_len,int(start_E0),int(load_time_limit), 100)
        #     H_load = np.load(load_live_file_path)
        # except:
        #     pass
        #     load_live_file_path = "/home/cug/jf330/Paper2/paper_results_gilda/TEST/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.npy".format(pre_syn, threshold, chain_len, int(start_E0), int(load_time_limit), 150)
        #     H_load = np.load(load_live_file_path)

        for i in range(0, pre_syn):
            spec_mol["H+{}".format(i)] = H_load[i]
            print("H+{} loaded value: {}".format(i, H_load[i]))

        live_file_path = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}.txt".format(pre_syn, threshold,
                                                                                                        chain_len,
                                                                                                        int(start_E0),
                                                                                                        int(time_limit),
                                                                                                        int(load_time_limit))
        H_file_path = path + "/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}_contFrom_{}".format(pre_syn, threshold, chain_len,
                                                                                                int(start_E0), int(time_limit),
                                                                                                        int(load_time_limit))
    else:

        live_file_path = path + "/live_pre{}_threshold{}_chain{}_E0{}_tLimit_{}.txt".format(pre_syn, threshold,
                                                                                           chain_len,
                                                                                           int(start_E0),
                                                                                           int(time_limit))

        H_file_path = path + "/CRN_H_pre{}_threshold{}_chain{}_E0{}_tLimit_{}".format(pre_syn, threshold, chain_len,
                                                                                      int(start_E0), int(time_limit))

    open(live_file_path, 'w').close()

    ### 12_A
    features = [[('I0', 0.015712458083621678), ('I1', 0.020453385783666335)]]
    ### 12_A_shorter
    # features = [[('I0', 0.015712458083621678), ('I1', 0.017253385783666335)]]
    ### 12_A_longer
    # features = [[('I0', 0.015712458083621678), ('I1', 0.023453385783666335)]]

    ### 123_A
    # features = [[('I0', 0.015712458083621678), ('I1', 0.020453385783666335), ('I2', 0.025453385783666335)]]
    ### 1234_A
    # features = [[('I0', 0.015712458083621678), ('I1', 0.020453385783666335), ('I2', 0.025453385783666335), ('I3', 0.028953385783666335)]]
    ### 12345_A
    # features = [[('I0', 0.015712458083621678), ('I1', 0.020453385783666335), ('I2', 0.025453385783666335), ('I3', 0.028953385783666335), ('I4', 0.030953385783666335)]]
    ### 123_A_more
    # features = [[('I0', 0.015712458083621678), ('I1', 0.018553385783666335), ('I2', 0.022453385783666335)]]

    print("Feature: {}".format(features))

    ## Simulate CRN LIVE
    nrm = NRM_live_test(pre_syn, spec_mol, reactions, threshold, features[0], dt=dt, mol_spike=mol_spike,
                   live_file_path=live_file_path, reporting=reporting, key_input=key_input, pattern_type=pattern_type)

    nrm.simulate(H_file_path, I_species, time_limit)


def main(path, test_type, chain, E0, part_run, parts_all, theta, volume, pattern_type, st_ratio):
    start_time = time.time()

    if test_type == "assoc_run":
        assoc_run(path, chain_len=chain, start_E0=E0, threshold=theta, volume=volume, pattern_type=pattern_type)
    if test_type == "train_STDP_properHill_nonpatterns_quick":
        train_STDP_properHill_nonpatterns_quick(path, chain_len=chain, start_E0=E0, threshold=theta, volume=volume, pattern_type=pattern_type, st_ratio=st_ratio)
                


    print("-- {} seconds, {} hours --".format((time.time() - start_time),(time.time() - start_time)/(60*60)))


if __name__ == '__main__':
    text_type = str
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--path',
        default='default',
        type=text_type,
        help='Default directory path',
    )

    parser.add_argument(
        '--test_type',
        default='test type',
        type=text_type,
        help='Test type',
    )

    parser.add_argument(
        '--chain',
        default=5,
        type=float,
        help='Chain len',
    )

    parser.add_argument(
        '--E0',
        default=5,
        type=int,
        help='E0 start',
    )

    parser.add_argument(
        '--part_run',
        default=1,
        type=int,
        help='Part to run',
    )

    parser.add_argument(
        '--parts_all',
        default=5,
        type=int,
        help='Parts all',
    )

    parser.add_argument(
        '--theta',
        default=0,
        type=int,
        help='Threshold',
    )

    parser.add_argument(
        '--volume',
        default=1,
        type=float,
        help='Volume',
    )

    parser.add_argument(
        '--pattern_type',
        default="temporal",
        type=text_type,
        help='Pattern type',
    )

    parser.add_argument(
        '--st_ratio',
        default=1,
        type=float,
        help='Spike threshold ratio',
    )


    args, unknown = parser.parse_known_args()
    try:
        main(args.path, args.test_type, args.chain, args.E0, args.part_run, args.parts_all, args.theta, args.volume, args.pattern_type, args.st_ratio)
    except KeyboardInterrupt:
        pass
    finally:
        print()
