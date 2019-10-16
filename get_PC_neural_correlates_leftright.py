import os
import csv
from matplotlib import pyplot as plt
import numpy as np
from numpy import trapz
from gen_regressor import *
# from sklearn import linear_model
import scipy.stats
from tools2getInt import *
from matplotlib.ticker import MaxNLocator


import pickle
plt.style.use(['dark_background'])
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

#expt = ['2 vs 6', '1 dot', '2 vs 3', '2 vs 4']
#expt = ['effect of target to competitor']
#expt = ['2 vs 3', '2 vs 4', '2 vs 6']
expt = ['2p_left_right_analysis']

fps = 7.65

labels_per_expt = []
all_dF_per_expt = []
all_xy_per_expt = []
percent_per_expt = []
ncells_per_expt = []
avg_ncells_expt = []
avg_percent_expt = []
PC_activity_per_expt = []
expts = []
cell_position = []
count = -1
all_diff = []
all_diff2 = []
maindir = 'E:\\new folder\\002 ANALYSIS\\'
#maindir = 'D:\\Semmelhack lab\\002 ANALYSIS\\2p_2dots\\brain imaging\\'
#maindir = 'D:\\Semmelhack lab\\002 ANALYSIS\\same fish_different competition\\brain imaging\\'
region = 'PVN'
for xp in expt:
    dir_xp = maindir + xp + '\\'
    dir_fish = next(os.walk(dir_xp))[1]
    #print dir_fish
    #dir_fish = [dir_fish[1],dir_fish[3], dir_fish[5],dir_fish[6],dir_fish[7],dir_fish[11]]
    labels_per_fish = []
    all_dF_per_fish = []
    all_xy_per_fish = []
    ncells_per_fish = []
    percent_per_fish = []
    PC_activity_per_fish = []
    all_neuron_activity_per_fish = []

    files = []
    #pc_info1 = get_pc_info(dir_xp + 'Summary of 1st PC_1dot.csv') # read the prey capture onset, and prey side
    #pc_info2 = get_pc_info(dir_xp + 'Summary of 1st PC_2vs3.csv')
    pc_info = get_pc_info(dir_xp + 'Summary of all 1st PC.csv')
    count_fish = 0
    print "EXPERIMENT: ", xp
    nth_fish = -1
    for fish in dir_fish:

        nth_fish +=1 # count how many fish have been process
        '''
        count_fish += 1
        if count_fish > 1:
            continue
        '''
        print "FISH: ", fish
        labels_per_trial = []
        all_dF_per_trial = []
        all_xy_per_trial = []
        ncells_per_trial = []
        percent_per_trial = []
        # dir_trials = next(os.walk(dir_xp + fish + '\\2deg vs 3deg\\avi\\imaging\\TIF\\moco\\'))[1]
        dir_trials = next(os.walk(dir_xp + fish + '\\reg\\avg_group\\' + region + '\\'))[1]
        # read the label information
        label_info = get_morphology(dir_xp + fish + '\\reg\\avg_group\\' + region + '\\label-Morphometry.csv')
        midline = get_midline(dir_xp + fish + '\\reg\\avg_group\\' + region + '\\Midline.txt')
        count_trial = 0
        PC_activity_per_trial = []
        PC_diff_per_trial = []
        all_neuron_activity_per_trial = []


        for trial in dir_trials:

            if "Mask" in trial:
                continue

            if 'L_' in trial or 'R_' in trial:  # FIND WHETHER THE PREY IS ON THE LEFT OR RIGHT
                pc_info = pc_info
            else:
                continue

            '''
            count_trial += 1
            if count_trial > 1:
                continue
            '''

            count += 1
            labels = []
            all_dF = []
            all_xy = []

            dir_input = dir_xp + fish + '\\reg\\avg_group\\' + region + '\\' + trial

            ts = 0
            if nth_fish >= 10:
                ts = 1
            curr_trial = trial.split('_')[ts] + '_' + trial.split('_')[-1] # store the current trial
            index = list(np.array(pc_info)[:,0]).index(curr_trial) # index of the current fish to find the onset
            index2 = index + 1
            index3 = index + 2

            #prey_size2 = pc_info[index2][3]
            #if int(prey_size2) == 2:
            #    print "PREY SELECTED by :", pc_info[index][0], "IS ", int(prey_size2)
            #    continue

            onset = split_convert2num(pc_info[index][2])
            onset = [match_2pfps(frame, 300.0, fps) for frame in onset]

            onset2 = split_convert2num(pc_info[index2][2])
            onset2 = [match_2pfps(frame, 300.0, fps) for frame in onset2]

            onset3 = split_convert2num(pc_info[index3][2])
            onset3 = [match_2pfps(frame, 300.0, fps) for frame in onset3]

            print curr_trial, index, pc_info[index][0], pc_info[index2][0]
            prey_side = pc_info[index][1]
            prey_side2 = pc_info[index2][1]
            prey_side3 = pc_info[index3][1]

            print "TRIAL: ", trial
            print "PC info: ", pc_info[index], pc_info[index2], pc_info[index3]
            if prey_side == 'r':
                prey_side = 'r'
            elif prey_side == 'l':
                prey_side = 'l'

            print "Prey side: ", prey_side

            # READ THE INTENSITIES

            dir_input2 = [dir_xp + fish + '\\reg\\avg_group\\' + region + '\\' + n for n in dir_trials if pc_info[index2][0].split('_')[0] in n and pc_info[index2][0].split('_')[1] in n]
            dir_input3 = [dir_xp + fish + '\\reg\\avg_group\\' + region + '\\' + n for n in dir_trials if pc_info[index3][0].split('_')[0] in n and pc_info[index3][0].split('_')[1] in n]

            mean_intensities = get_allInt(dir_input)
            mean_intensities2 = get_allInt(dir_input2[0])
            mean_intensities3 = get_allInt(dir_input3[0])

            # GENERATE PREY CAPTURE REGRESSOR
            offset = -8 # move up or down from PC onset
            offset2 = 8 # move up or down from PC onset

            duration = int(1*fps)
            stim = gen_stim_onset(len(mean_intensities), range(onset[0] + offset, onset[0] + offset + duration), 12)

            norm = float(max(stim))
            norm_stim = [d / norm for d in stim]

            framet = 1.0 / fps
            time = [t * framet for t in range(len(mean_intensities))]
            waiting_time = (1.5*fps)
            # time points to measure the correlation between stimulus regressors and activity
            t0_of_interest = int(onset[0] + offset)
            t1_of_interest = int(onset[0])

            t0_of_interest2 = int(onset2[0] + offset)
            t1_of_interest2 = int(onset2[0])

            t0_of_interest3 = int(onset3[0] + offset)
            t1_of_interest3 = int(onset3[0])

            t0_onset = int(onset[0] - offset2)
            t1_onset = int(onset[0] + offset2)

            t0_onset2 = int(onset2[0] - offset2)
            t1_onset2 = int(onset2[0] + offset2)

            t0_onset3 = int(onset3[0] - offset2)
            t1_onset3 = int(onset3[0] + offset2)

            '''
            plt.plot(time,norm_stim, c=[0,1,1], linewidth=3)
            plt.plot(time[t0_of_interest:t1_of_interest],norm_stim[t0_of_interest:t1_of_interest], c=[0,1,0], linewidth=3)
            plt.ylabel("$\Delta$F/F$_o$", fontsize=20)
            plt.xlabel("Time (s)", fontsize=20)
            plt.show()
            '''
            c = []
            total_cells = 0
            anterior = 0
            posterior = 0
            PC_activity_per_neuron = []
            PC1 = []
            PC2 = []
            PC3 = []

            PC_diff_per_neuron = []
            all_neuron_activity_per_neuron = []


            for i in range(len(np.array(mean_intensities)[1, :])):

                f1 = np.array(mean_intensities)[:, i]
                f2 = [f3[1] for f3 in f1]
                f = butter_freq_filter(f2, 3.5, fps, 3)  # low pass filter of left eye movements
                base = np.percentile(f, 25)

                # find the corresponding 2 dots
                f1_2dots = np.array(mean_intensities2)[:, i]
                f2_2dots = [f3[1] for f3 in f1_2dots]
                f_2dots = butter_freq_filter(f2_2dots, 3.5, fps, 3)  # low pass filter of left eye movements
                base_2dots = np.percentile(f_2dots, 25)

                # find the corresponding 2 dots
                f1_2dots2 = np.array(mean_intensities3)[:, i]
                f2_2dots2 = [f3[1] for f3 in f1_2dots2]
                f_2dots2 = butter_freq_filter(f2_2dots2, 3.5, fps, 3)  # low pass filter of left eye movements
                base_2dots2 = np.percentile(f_2dots2, 25)

                all_base = np.percentile([f, f_2dots, f_2dots2], 25)
                # find which label, and find the area
                curr_lbl = f1[0][0]
                indx_label = list(np.array(label_info)[:,0]).index(curr_lbl)
                lbl_area = label_info[indx_label][1] # the area of the current label
                lbl_circ = label_info[indx_label][3]  # the area of the current label
                xy_lbl = [label_info[indx_label][4], label_info[indx_label][5]] # the area of the current label

                #if prey_side == 'r':
                #    if xy_lbl[0] > midline[0]: # skip it if the neuron is on the right tectum
                #        continue
                #elif prey_side == 'l':
                #    if xy_lbl[0] < midline[0]: # skip it if the neuron is on the left tectum
                #        continue

                if lbl_area > 200.0:
                    continue
                if lbl_area < 15.0:
                    continue

                if lbl_circ < 0.5:
                    continue
                if lbl_circ > 1.5:
                    continue

                total_cells += 1
                # get the delta f
                dF = []
                dF_2dots = []
                dF_2dots2 = []

                for j, val in enumerate(f):
                    #dF.append((val - all_base) / np.abs(float(all_base)))
                    #dF_2dots.append((f_2dots[j] - all_base) / np.abs(float(all_base)))
                    #dF_2dots2.append((f_2dots2[j] - all_base) / np.abs(float(all_base)))
                    dF.append((val - base) / np.abs(float(base)))
                    dF_2dots.append((f_2dots[j] - base_2dots) / np.abs(float(base_2dots)))
                    dF_2dots2.append((f_2dots2[j] - base_2dots2) / np.abs(float(base_2dots2)))

                all_neuron_activity_per_neuron.append([dF[t0_onset:t1_onset], dF_2dots[t0_onset2:t1_onset2], dF_2dots2[t0_onset3:t1_onset3]])
                if np.mean(dF_2dots[t0_of_interest2:t1_of_interest2]) < 2.0*np.std(dF_2dots):
                    continue
                #corr = scipy.stats.pearsonr(dF[t0_of_interest:t1_of_interest], norm_stim[t0_of_interest:t1_of_interest])
                #if corr[0] < 0.5 or np.isnan(corr[0]):
                #    continue

                #if xy_lbl[1] > midline[1]:
                #    anterior += 1
                #elif xy_lbl[1] < midline[1]:
                #    posterior += 1

                c.append(f)
                all_dF.append(dF)
                all_xy.append([xy_lbl, midline[1]])
                labels.append(int(f1[0][0]))

                PC_activity_per_neuron.append([dF[t0_of_interest:t1_of_interest], dF_2dots[t0_of_interest2:t1_of_interest2],  dF_2dots2[t0_of_interest3:t1_of_interest3], int(f1[0][0])])
                PC_diff_per_neuron.append(np.subtract(dF[t0_of_interest:t1_of_interest], dF_2dots[t0_of_interest2:t1_of_interest2]))
                '''
                PC1.append(dF[t0_of_interest:t1_of_interest])
                PC2.append(dF_2dots[t0_of_interest2:t1_of_interest2])
                PC3.append(dF_2dots2[t0_of_interest3:t1_of_interest3])
                plt.plot(time, dF, c=[0, 1, 0], linewidth=3, alpha = 0.5)
                plt.plot(time, dF_2dots, c=[0, 1, 1], linewidth=3, alpha = 0.5)
                plt.plot(time, dF_2dots2, c=[1, 0, 1], linewidth=3, alpha = 0.5)
                plt.plot(time, norm_stim)
                plt.plot(time[t0_of_interest:t1_of_interest], dF[t0_of_interest:t1_of_interest], c=[0, 1, 0], linewidth=3)
                plt.plot(time[t0_of_interest:t1_of_interest], dF_2dots[t0_of_interest:t1_of_interest], c=[0, 1, 1], linewidth=3)
                plt.plot(time[t0_of_interest:t1_of_interest], dF_2dots2[t0_of_interest:t1_of_interest], c=[1, 0, 1], linewidth=3)

                plt.ylabel("$\Delta$F/F$_o$", fontsize=20)
                plt.xlabel("Time (s)", fontsize=20)
                plt.show()
                #stop
                '''
            #cell_position.append(float(anterior)/(posterior+anterior))
            ncells_per_trial.append(len(c))
            percent_per_trial.append((len(c) / float(total_cells)) * 100.0)
            print "Trial: ", trial, onset, duration
            print len(c), total_cells
            print len(c) / float(total_cells) * 100.0

            # print np.mean(c, axis = 0)
            # plt.plot(np.mean(c, axis = 0))
            #diff = np.subtract(np.mean(np.array(PC_activity_per_neuron)[0],0) - np.mean(np.array(PC_activity_per_neuron[1]),0))
            #diff2 = np.subtract(np.mean(PC1,0) - np.mean(PC2,0))
            #all_diff.append(diff2)
            '''
            plt.plot(range(t0_of_interest,t1_of_interest), np.mean(PC1,0), c=[0, 1, 0], linewidth=3)
            plt.plot(range(t0_of_interest,t1_of_interest), np.mean(PC2,0), c=[0, 1, 1], linewidth=3)

            plt.ylabel("$\Delta$F/F$_o$", fontsize=20)
            plt.xlabel("Time (s)", fontsize=20)
            plt.show()
            '''
            all_neuron_activity_per_trial.append(all_neuron_activity_per_neuron)
            PC_diff_per_trial.append(PC_activity_per_neuron)
            all_diff2.append([PC1, PC2, PC3])
            all_diff.append([np.mean(PC1,0), np.mean(PC2,0),  np.mean(PC3,0)])
            PC_activity_per_trial.append(PC_activity_per_neuron)
            all_dF_per_trial.append(all_dF)
            all_xy_per_trial.append(all_xy)
            labels_per_trial.append(labels)

        all_neuron_activity_per_fish.append(all_neuron_activity_per_trial)
        PC_activity_per_fish.append(PC_activity_per_trial)
        ncells_per_fish.append(ncells_per_trial)
        percent_per_fish.append(percent_per_trial)
        labels_per_fish.append(labels_per_trial)
        all_dF_per_fish.append(all_dF_per_trial)
        all_xy_per_fish.append(all_xy_per_trial)
    '''
    summary_per_fish = {"trial": trial, "labels": labels_per_fish,
               "df": all_dF_per_fish, "xy": all_xy_per_fish,
               "PC_activity_per_fish": PC_activity_per_fish,"PC_activity_per_trial": PC_activity_per_trial ,"ncells": ncells_per_fish, "percent": percent_per_fish}

    pickle_out1 = open(dir_xp + xp + '_summary_analysis_3s_v2.pickle', 'w')
    pickle.dump(summary_per_fish, pickle_out1)
    pickle_out1.close()
    '''
    expts.append(xp)
    PC_activity_per_expt.append(PC_activity_per_fish)
    ncells_per_expt.append(ncells_per_fish)
    percent_per_expt.append(percent_per_fish)
    labels_per_expt.append(labels_per_fish)
    all_dF_per_expt.append(all_dF_per_fish)
    all_xy_per_expt.append(all_xy_per_fish)

    #avg_percent_expt.append([np.mean(percent_per_expt), np.std(percent_per_expt)/(np.sqrt(len(percent_per_expt)))])
    #avg_ncells_expt.append([np.mean(ncells_per_expt), np.std(ncells_per_expt)/(np.sqrt(len(ncells_per_expt)))])
    '''
    all_right = []

    for xp in range(len(all_xy_per_expt)): # each depth
        right = []
        for trial in range(len(all_xy_per_expt[xp])): # each trial
            ri = 0
            le = 0
            for label in range(len(all_xy_per_expt[xp][trial])):  # each trial
                x = all_xy_per_expt[xp][trial][label][0]
                y = all_xy_per_expt[xp][trial][1]

                if x >= 270.0:
                    ri += 1
                else:
                    le += 1

            right.append(ri/float(ri+le))
        #plt.show()
        print "TRIAL:", all_xy_per_expt[xp][trial]
        print right
        all_right.append([np.mean(right), np.std(right)/np.sqrt(len(right))])
    '''
#print all_right
print "DONE"
print expts
#print labels_per_expt
#print all_dF_per_expt
#print all_xy_per_expt
print avg_percent_expt
print avg_ncells_expt


summary = {"all_activity_per_fish": all_neuron_activity_per_fish,"expt": expts, "labels_per_expt": labels_per_expt, 'all_diff': all_diff,
           "all_df_per_expt": all_dF_per_expt, "all_xy_per_expt": all_xy_per_expt,
           "PC_activity": PC_activity_per_expt, "PC_activity_per_fish": PC_activity_per_fish,"PC_activity_per_trial": PC_activity_per_trial, "ncells": ncells_per_expt, "percent": percent_per_expt}

pickle_out = open(maindir +'summary_LRanalysis_ALLSTD_diffbase_2std_1sb4PC_2ndcase_' + region + '.pickle', 'w')
pickle.dump(summary,pickle_out)
pickle_out.close()

'''
dF_per_expt = []
count = -1
for xp in range(len(all_df_per_expt)): # each depth
    dF_per_trial = []
    for trial in range(len(all_df_per_expt[xp])): # each trial
        count += 1
        onset = int(condi[count][0] * fps)
        duration = int(condi[count][1] * fps)
        t0_of_interest = int(onset)-20
        t1_of_interest = int(onset+duration)+39
        for label in range(len(all_df_per_expt[xp][trial])):  # each trial
            dF_per_trial.append(all_df_per_expt[xp][trial][label][t0_of_interest:t1_of_interest])
    dF_per_expt.append([np.mean(dF_per_trial,0), np.std(dF_per_trial,0)])

print len(dF_per_expt)
fig, ax= plt.subplots(1, 1, figsize=(20, 10))

for i in range(len(dF_per_expt)):
    if i == 0:
        color = [0, 1, 0]
    elif i == 1:
        color = [1, 0, 1]
    time = [t * framet for t in range(len(dF_per_expt[i][0]))]
    ax.plot(time, dF_per_expt[i][0],c=color)
    ax.fill_between(time, dF_per_expt[i][0]-dF_per_expt[i][1],
                     dF_per_expt[i][0]+dF_per_expt[i][1], alpha=0.3, facecolor=color)

ax.axvline(1,c=[1,1,1], ls='--')
ax.yaxis.set_major_locator(MaxNLocator(6))
#ax.xaxis.set_major_locator(MaxNLocator(4))

ax.set_ylabel('$\Delta$F/F$_o$', fontsize=30)
ax.set_xlabel('Time (s)', fontsize=30)
ax.set_ylim([-0.1,0.6])
fig.savefig("C:\\Users\\Semmelhack Lab\\Documents\\Analysis\\2p analysis\\2dotsPC2.png")

plt.show()
'''


