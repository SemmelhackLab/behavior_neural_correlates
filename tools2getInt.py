from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from numpy import trapz
import pickle, csv
from scipy.signal import butter, filtfilt, savgol_filter
import os
dir_input = "D:\\Trial\\TIF\\"
file1 = "2P008-A1_3deg@40ds-140_RL-LR_2dots-10-70_70-10_70um_BLANK.tif"
file2 = "2P008-A1_3deg@40ds-140_RL-LR_2dots-10-70_70-10_70um_T31.tif"


def butter_filter(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_freq_filter(data, cutoff, fs, order=5):
    b, a = butter_filter(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def tif2array(img):
    '''
    :param img: the tiff file time lapse you want to convert to array
    :return: a 3D array
    '''
    tiff = Image.open(img)
    im = []

    for i in range(tiff.n_frames):
        tiff.seek(i)
        # print len(np.array(tiff))
        im.append(np.array(tiff))

    return im


def get_baseline(im):
    '''
    :param im: the array that contains the time lapse series
    :return: the average of the array time lapse or mean intensity of the time series
    '''
    im = im[0:20]
    baseline = np.mean(im, 0)

    return baseline


def get_pixelint(baseline, im):
    n_frames = len(im)
    deltaFoF = []
    for i in range(n_frames):
        deltaF = np.subtract(im[i], baseline)
        deltaFoF.append(np.divide(deltaF, baseline))  # deltaF/F)
        # plt.imshow(deltaFoF[i], cmap='gray')
        # plt.show()

    return deltaFoF


def get_morphology(label_morph):
    label_info = []
    with open(label_morph) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        headers = readCSV.next()
        lbl = headers.index('Label')
        area = headers.index('Area')
        peri = headers.index('Perimeter')
        circ = headers.index('Circularity')
        xcoord = headers.index('Centroid.X')
        ycoord = headers.index('Centroid.Y')

        for row in readCSV:
            label_info.append([int(row[lbl]), float(row[area]),
                               float(row[peri]), float(row[circ]),
                               float(row[xcoord]), float(row[ycoord])])

    return label_info


def get_midline(input):

    file = open(input,"r")
    midline = file.readlines()
    midline = [int(midline[1].split()[0]),
               int((int(midline[2].split()[1]) - int(midline[1].split()[1]))/2)] # 1st element is the axis for separating left and right, second one is the line separating anterior and posterior

    file.close()

    return midline


def get_pc_info(input):

    pc_data = []

    with open(input) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        headers = readCSV.next()
        trial = headers.index('Fish')
        prey = headers.index('Prey Side')
        dur = headers.index('Duration')
        size = headers.index('Prey Size')

        for row in readCSV:
            pc_data.append([row[trial], row[prey], row[dur], row[size]])

    return pc_data


def get_allInt(dir_input):
    files = []
    for file in os.listdir(dir_input):  # read files with .csv then store the filename to files
        if file.endswith(".csv"):
            files.append(file)

    files.sort(key=lambda f: int(filter(str.isdigit, f)))  # sort the files in ascending order
    mean_intensities = []

    for file in files:
        inten = []
        with open(dir_input + '\\' + file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            headers = readCSV.next()
            mean_col = headers.index('Mean')
            lbl = headers.index('Label')

            for row in readCSV:
                inten.append([int(row[lbl]), float(row[mean_col])])

        mean_intensities.append(inten)

    return mean_intensities


def split_convert2num(string_array):

    t0 = int(string_array.split()[0].replace('[', '').replace(',', ''))
    t1 = int(string_array.split()[1].replace(']', ''))

    return [t0, t1]


def match_2pfps(frame, behave_fps, brain_fps):

    frame2p = int(frame/float(behave_fps) * float(brain_fps))

    return frame2p
'''
baseline = get_baseline(tif2array(dir_input+file2))
plt.imshow(baseline, cmap='gray')
plt.show()
im = tif2array(dir_input+file2)

d = get_pixelint(baseline, im)

cell = []
#print len(np.array(im)[0][:,511])
for i in range(512):
    for j in range(512):
        cellt = []
        for k in range(len(im)):
            #plt.imshow(d[k], cmap = 'gray')
            #plt.show()
            cellt.append(d[k][i,j])
        cell.append(cellt)

area = []
for i in range(len(cell)):
    if trapz(cell[i]) > 500:
        area.append(cell[i])
        #plt.plot(cell[i])
        #plt.show()

print len(area)
'''

'''
with open('cell.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump(im, f)
'''

# print len(area)
# plt.plot(cell)
# plt.show()
# a = np.subtract(d[250], d[0])
# plt.imshow(d[396], cmap ='gray')
# plt.show()
# print
# print d[395][:][1].shape
# plt.plot(np.array(d)[:])
# plt.show()