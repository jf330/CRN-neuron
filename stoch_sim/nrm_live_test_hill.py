import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import utils.datamaker
from collections import deque
import sys
import gzip


class NRM_live_test_HILL:

    def __init__(self, pre_syn, spec_mol, reactions, threshold, feature=[], dt=1000, mol_spike=100, live_file_path="/Users/jf330/Desktop", reporting=True, key_input=True, pattern_type="temporal", steepness=20):

        self.spec_mol = spec_mol
        self.reactions = reactions
        self.pre_syn = pre_syn
        self.dt = dt
        self.entropy = 0

        self.reporting = reporting
        self.key_input = key_input
        self.key_input = key_input

        self.t = 0
        self.report_number = 1

        # self.report_interval = 0.01 / self.dt
        # self.report_interval = 0.5 / self.dt
        # self.report_interval = 0.3 / self.dt
        # self.report_interval = 0.5 / self.dt
        # self.report_interval = 1 / self.dt
        self.report_interval = 0  ### ALWAYS REPORT

        self.pattern_type = pattern_type
        self.live_file_path = live_file_path

        self.report_save_interval = 20000
        # self.report_save_interval = 1000
        # self.report_save_interval = 1

        self.values_string = ""
        self.f = open(self.live_file_path, "a+")
        # self.f = gzip.open(self.live_file_path, "a+")

        # self.mol_hist = []
        # self.time_hist = []
        # self.prop_hist = []

        self.mol_spike = mol_spike
        self.K = threshold
        # self.h = 500  # Hill functions steepness
        # self.h = 200  # Hill functions steepness
        # self.h = 150  # Hill functions steepness
        # self.h = 100  # Hill functions steepness
        # self.h = 50  # Hill functions steepness
        # self.h = 20  # Hill functions steepness
        # self.h = 10  # Hill functions steepness

        self.h = steepness  # Hill functions steepness

        self.noisy_event_q = deque()
        self.pattern_event_q = deque()
        self.fea = feature

        self.total_spikes = {}

        self.fea_t = 0
        self.fea_start_t = 0
        self.fea_idx = 0
        self.fea_flag = 0

        self.taus = np.zeros((len(reactions)), dtype=float)
        self.props = np.zeros((len(reactions)), dtype=float)

    def calc_taus(self):
        randoms = np.random.random(len(self.reactions))  # Regenerate for each time step M random numbers
        randoms_log = np.log(randoms) * -1

        for idx in range(0, len(self.reactions)):  # Calculate propensities
            left = self.reactions[idx][0]  # Reactants
            # right = self.reactions[idx][1]  # Products
            k = self.reactions[idx][2][0]  # Rate constant k
            # pair = self.reactions[idx][2][1]  # Reverse reaction idx

            if k > 0:
                prop = k
            # else:
            elif k == -1:
                ### Threshold (self.K) is a Hill function coefficient
                # prop = (((self.spec_mol.get(left[0][0]) ** self.h) / (self.K ** self.h + self.spec_mol.get(left[0][0]) ** self.h)) * self.dt)
                prop = (((self.spec_mol.get(left[0][0]) ** self.h) / (self.K ** self.h + self.spec_mol.get(left[0][0]) ** self.h)) * self.dt) * 10
                # prop = (((self.spec_mol.get(left[0]) ** self.h) / (self.K ** self.h + self.spec_mol.get(left[0]) ** self.h)) * 10 * self.dt * self.h)
                # prop = 0

            try:
                for l in left:  # Consider all reactants
                    if self.spec_mol.get(l[0]) >= l[1]:
                        prop = prop * self.spec_mol.get(l[0])
                        # prop = prop * self.spec_mol.get(l[0]) * l[1]
                    else:
                        prop = 0
            except:
                print("Exception in propensity calculation species: {}".format(left))
                break

            self.props[idx] = prop
            if prop == 0:
                self.taus[idx] = np.inf
            else:
                self.taus[idx] = randoms_log[idx] / prop

        tau_eta = min(self.taus)
        idx_tau_eta = np.where(self.taus == tau_eta)[0][0]

        return tau_eta, idx_tau_eta

    def execute_react(self, idx_tau_eta):
        ###  Execute reaction
        left = self.reactions[idx_tau_eta][0]  # Reactants
        right = self.reactions[idx_tau_eta][1]  # Products
        k = self.reactions[idx_tau_eta][2][0]  # Rate constant k
        pair = self.reactions[idx_tau_eta][2][1][0]  # Reverse reaction idx

        for l in left:
            # if l[0] != "NULL" and "g" in l[0] == False:
            if l[0] != "NULL":
                self.spec_mol[l[0]] = self.spec_mol[l[0]] - l[1]

        for r in right:
            # if r[0] != "NULL" and "g" in r[0] == False:
            if r[0] != "NULL":
                self.spec_mol[r[0]] = self.spec_mol[r[0]] + r[1]

        ### Calculate Entropy
        # k_rev = self.reactions[idx_tau_eta][2][0]  # Reverse rate constant k_rev
        # prop_back = k_rev
        # for l_b in self.reactions[pair][0]:
        #     prop_back = prop_back * self.spec_mol.get(l_b[0])

        # self.entropy = self.entropy + np.log(self.props[idx_tau_eta]/prop_back)
        # self.entropy = self.entropy - np.log(self.props[idx_tau_eta]/prop_back)
        # self.entropy = self.entropy + np.log(-self.props[idx_tau_eta]/prop_back)

        # print("Entropy prod. at time {} = {}".format(self.t, self.entropy))

    def step(self):

        ### Calculate time of next reaction
        tau_eta, idx_tau_eta = self.calc_taus()

        next_pattern_t = self.pattern_event_q[0][1]
        next_random_t = self.noisy_event_q[0][1]

        do_pattern = False
        if next_pattern_t < next_random_t:
            do_pattern = True

        ### Check if fixed time event should happen
        if do_pattern:
            if self.t + tau_eta > self.pattern_event_q[0][1] >= self.t:

                self.spec_mol[self.pattern_event_q[0][0]] += self.mol_spike
                # self.total_spikes[self.pattern_event_q[0][0]] += 1

                ###  Increment time
                tau_eta = self.pattern_event_q[0][1] - self.t
                self.fea_flag = self.pattern_event_q[0][2]

                self.pattern_event_q.popleft()
            else:
                ### Execute reaction
                self.execute_react(idx_tau_eta)
        else:
            if self.t + tau_eta > self.noisy_event_q[0][1] >= self.t:

                self.spec_mol[self.noisy_event_q[0][0]] += self.mol_spike
                # self.total_spikes[self.noisy_event_q[0][0]] += 1

                ###  Increment time
                tau_eta = self.noisy_event_q[0][1] - self.t
                self.fea_flag = self.noisy_event_q[0][2]

                self.noisy_event_q.popleft()
            else:
                ### Execute reaction
                self.execute_react(idx_tau_eta)

        ### Check if report should be made
        if self.reporting:
            ### Always report
            if self.report_interval == 0:
                self.make_report()
                self.report_number += 1
            else:
                ### Sample reports
                if self.t + tau_eta > self.report_number * self.report_interval >= self.t:
                    self.make_report()
                    if self.t + tau_eta > (self.report_number + 1) * self.report_interval:
                        # print("Skipped report at {}".format(self.report_number))
                        self.report_number = int((self.t + tau_eta) / self.report_interval) + 1
                        # print("Next report at {}".format(self.report_number))
                    else:
                        self.report_number += 1
                    # self.write_report()

        ### Increment time
        if tau_eta != np.inf:
            self.t = self.t + tau_eta
        else:
            raise ValueError('INF tau_eta!')

        if self.report_number % self.report_save_interval == 0:
            self.write_report()

    def cut_file(self, path, interval):
        with open(path, 'r') as f_in:
            data = f_in.read().splitlines(True)
        with open(path, 'w') as f_out:
            f_out.writelines(data[interval:])

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def make_report(self):
        ### SAMPLE FOR LATER USE SPECIES DATA
        # print("Report at time: {}".format(self.t))
        # self.spec_mol["R_fast"] = self.entropy

        ### Get current molecular composition
        values = list(self.spec_mol.values())

        ### Save all data to file
        for i in range(0, values.__len__()):
            try:
                self.values_string += str(int(values[i])) + ", "
            except:
                print("Exception in species mol: {}".format(self.spec_mol))

        self.values_string += str(self.t) + ", "
        self.values_string += str(self.fea_flag) + "\n"

    def write_report(self):
        print("Writing report at {}, report number: {}".format(self.t, self.report_number))
        # self.f = gzip.open(self.live_file_path, "a+")
        # self.f = open(self.live_file_path, "a+")

        # self.f.write(self.values_string.encode('utf-8'))
        self.f.write(self.values_string)
        self.values_string = ""
        # self.f.close()

    def simulate(self, H_file_path, I_species, time_limit):
        pattern_noise = 0.000001 ### original
        # pattern_noise = 0 ### for fig 7
        # pattern_prob = 0.5

        if self.key_input:
            from kbit import KBHit
            kb = KBHit()
            print('Hit any number key <= pre_syn to add input, or ESC to exit')

        input_rate = 2 / (self.dt / 1000)
        # input_rate = 3 / (self.dt / 1000)
        # input_rate = 4 / (self.dt / 1000)

        pattern_rate = 2 / (self.dt / 1000)

        # input_rate = 0.000001 / (self.dt / 1000)
        # pattern_rate = 0.000001 / (self.dt / 1000)

        # for i in range(0, self.pre_syn):
        #     self.total_spikes["I{}".format(i)] = 0

        # self.noisy_event_q.append(["I0", 0.02, 0])
        # self.noisy_event_q.append(["I4", 0.03, 0])
        # self.noisy_event_q.append(["I4", 0.04, 0])
        # self.noisy_event_q.append(["I5", 0.29, 0])
        #
        # self.noisy_event_q.append(["I2", 0.68, 0])
        # self.noisy_event_q.append(["I3", 0.77, 0])
        #
        # self.noisy_event_q.append(["I0", 0.87, 0])
        # self.noisy_event_q.append(["I0", 1.1, 0])

        # self.noisy_event_q.append(["I0", 2000.0, 0])
        # self.pattern_event_q.append(["I0", 2000.0, 0])

        if self.pattern_type != "temporal":
            self.pattern_event_q.append(["I0", 2000.0, 1])
            # input_rate = 1.5 / (self.dt / 1000)
        # else:
        #     self.noisy_event_q.append(["I0", 1000.0, 1])

        counter = 1
        limit = 1
        ### Time limit
        while self.t < time_limit:
            if self.t > limit * counter:
                counter += 1
                print("Spec mol: {} \n at t: {}".format(self.spec_mol, self.t))
            ### User input
            if self.key_input:
                if kb.kbhit():
                    c = kb.getch()
                    if c == 't':
                        print(" 'T': Current simulation time: {}".format(self.t))
                    elif c == 'f':
                        if self.pattern_event_q.__len__():
                            self.fea_t = self.t + random.expovariate(pattern_rate)
                            for i in range(0, len(self.fea)):
                                self.pattern_event_q.append([self.fea[i][0], self.fea[i][1] + self.fea_t, 1])
                                # if i != 0:
                                    # pattern_chance = np.random.rand()
                                    # if pattern_chance > pattern_prob:
                                    #     do
                        print(" 'F': Feature: {}".format(self.fea[0][1] + self.t))
                    elif c == 'h':
                        H_values = []
                        for i in range(0, self.pre_syn):
                            H_values.append(self.spec_mol["H+{}".format(i)])
                        print("H values: {}".format(H_values))
                    elif ord(c) == 27:  # ESC
                        print(" 'E': Exiting")
                        ### Save H+ values
                        H_values = []
                        for i in range(0, self.pre_syn):
                            H_values.append(self.spec_mol["H+{}".format(i)])

                        print("H values: {}".format(H_values))
                        np.save(H_file_path, H_values, allow_pickle=True)

                        sys.exit()
                    elif int(c) <= self.pre_syn:
                        event = ["I{}".format(int(c)), self.t, 1]
                        self.noisy_event_q.clear()
                        self.noisy_event_q.append(event)
                        print(event)
                    else:
                        print("Key {} pressed.".format(c))

            ### Feature input
            if self.pattern_type == "temporal":
                if self.pattern_event_q.__len__() == 0:
                    self.fea_t = self.t + random.expovariate(pattern_rate)

                    for i in range(0, len(self.fea)):
                        temp_noise = np.random.normal(0, pattern_noise)
                        cur_timing = self.fea[i][1] + self.fea_t + temp_noise
                        self.pattern_event_q.append([self.fea[i][0], cur_timing, 1])
                        # if i != 0:
                        #     pattern_rand = np.rand.random(0,1)

            ### Random input
            if self.noisy_event_q.__len__() == 0:
                self.noisy_event_q.append(utils.datamaker.poisson_one_input_live(I_species, self.t, input_rate))

            self.step()

        ### Make final report
        self.write_report()
        self.f.close()

        print(self.spec_mol)

        ### Save H+ values
        H_values = []
        for i in range(0, self.pre_syn):
            H_values.append(self.spec_mol["H+{}".format(i)])
        np.save(H_file_path, H_values, allow_pickle=True)
        print("H values: {}".format(H_values))
        # print(self.t)
        # print(self.total_spikes)

    def simulate_fixed(self, H_file_path, I_species, time_limit):
        pattern_noise = 0.000001

        if self.key_input:
            from kbit import KBHit
            kb = KBHit()
            print('Hit any number key <= pre_syn to add input, or ESC to exit')

        ### Association figure
        self.noisy_event_q.append(["I0", 0.01, 0])
        self.noisy_event_q.append(["I1", 0.1, 0])

        self.noisy_event_q.append(["I0", 0.25, 0])
        self.noisy_event_q.append(["I1", 0.251, 0])

        self.noisy_event_q.append(["I0", 0.35, 0])
        self.noisy_event_q.append(["I1", 0.351, 0])

        self.noisy_event_q.append(["I0", 0.45, 0])
        self.noisy_event_q.append(["I1", 0.451, 0])

        self.noisy_event_q.append(["I0", 0.55, 0])
        self.noisy_event_q.append(["I1", 0.551, 0])

        self.noisy_event_q.append(["I0", 0.65, 0])
        self.noisy_event_q.append(["I1", 0.651, 0])

        self.noisy_event_q.append(["I0", 0.8, 0])
        self.noisy_event_q.append(["I1", 0.9, 0])

        ### Learning figure
        # self.noisy_event_q.append(["I0", 0.01, 0])
        # self.noisy_event_q.append(["I0", 0.02, 0])
        # self.noisy_event_q.append(["I1", 0.1, 0])

        self.pattern_event_q.append(["I0", 1.7, 1])
        self.noisy_event_q.append(["I0", 1.7, 1])


        ### Time limit
        counter = 1
        limit = 0.5
        ### Time limit
        while self.t < time_limit:
            if self.t > limit * counter:
                counter += 1
                print("Spec mol: {} \n at t: {}".format(self.spec_mol, self.t))
            self.step()
        ### Make final report
        self.write_report()
        self.f.close()

        ### Save H+ values
        H_values = []
        for i in range(0, self.pre_syn):
            H_values.append(self.spec_mol["H+{}".format(i)])
        np.save(H_file_path, H_values, allow_pickle=True)
        print("H values: {}".format(H_values))
        print(self.t)
        # print(self.total_spikes)

    def simulate_lite(self, I_species, time_limit):
        input_rate = 1.5 / (self.dt / 1000)
        pattern_rate = 3 / (self.dt / 1000)

        self.fea_t = self.t + random.expovariate(pattern_rate)
        # self.fea_t = 1.5

        self.noisy_event_q.append(["I0", np.inf, 0])
        self.pattern_event_q.append(["I0", np.inf, 0])

        ### Time limit
        while self.t < time_limit:

            # ### Feature input
            # if self.fea_t <= self.t:
            #     self.noisy_event_q.clear()
            #     for i in range(0, len(self.fea)):
            #         self.noisy_event_q.append([self.fea[i][0], self.fea[i][1] + self.t, 1])
            #     self.fea_t = self.t + 0.05 + random.expovariate(pattern_rate)
            #
            # ### Random input
            # if self.noisy_event_q.__len__() == 0:
            #     self.noisy_event_q.append(utils.datamaker.poisson_one_input_live(I_species, self.t, input_rate))

            self.step()

        ### Save all data for later
        # self.prop_hist.append(list(self.props))
        # self.mol_hist.append(list(self.spec_mol.values()))
        # self.time_hist.append(self.t)

        # return self.prop_hist, self.mol_hist, self.time_hist
        # return self.mol_hist, self.time_hist

    def next_time(self, rate_param):
        return -math.log(1.0 - random.random()) / rate_param
