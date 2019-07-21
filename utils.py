import numpy as np
import os
import glob
import scipy.ndimage.interpolation as inter
import json
import math


class Action_Dataset():
    def __init__(self, dir, two_dimensions=True):
        print('loading data from:', dir)

        self.pose_paths = glob.glob(os.path.join(dir, '*', '*', '*', '*.json'))
        self.pose_paths.sort()
        self.twod = two_dimensions

    def read_json(self, pose_path):
        pose = []
        with open(pose_path) as json_file:
            data = json.load(json_file)
        for p in data['people']:
            pose = p['pose_keypoints_2d']
            if self.twod:
                for index in range(len(pose), 0, -1):
                    if index % 3 == 0:
                        del pose[index - 1]
            return pose

    def get_data(self, test_set_folder):
        cross_set = {}
        cross_set[0] = ['s11d1', 's11d2', 's11d3', 's11d4', 's12d1', 's12d2']
        cross_set[1] = ['s12d3', 's12d4', 's13d1', 's13d2', 's13d3', 's13d4']
        cross_set[2] = ['s14d1', 's14d2', 's14d3', 's14d4', 's15d1', 's15d2']
        cross_set[3] = ['s15d3', 's15d4', 's16d1', 's16d2', 's16d3', 's16d4']
        cross_set[4] = ['s17d1', 's17d2', 's17d3', 's17d4', 's18d1', 's18d2']
        cross_set[5] = ['s18d3', 's18d4', 's19d1', 's19d2', 's19d3', 's19d4']
        cross_set[6] = ['s20d1', 's20d2', 's20d3', 's20d4', 's21d1', 's21d2', 's21d3']
        cross_set[7] = ['s21d4', 's23d1', 's23d2', 's23d3', 's23d4', 's24d1', 's24d2']
        cross_set[8] = ['s24d3', 's24d4', 's25d1', 's25d2', 's25d3', 's25d4', 's01d1']
        cross_set[9] = ['s01d2', 's01d3', 's01d4', 's04d1', 's04d2', 's04d3', 's04d4']
        cross_set[10] = ['s22d1', 's22d2', 's22d3', 's22d4', 's02d1', 's02d2', 's02d3', 's02d4', 's03d1', 's03d2', 's03d3', 's03d4', 's05d1', 's05d2', 's05d3', 's05d4', 's06d1', 's06d2', 's06d3', 's06d4', 's07d1', 's07d2', 's07d3', 's07d4', 's08d1', 's08d2', 's08d3', 's08d4', 's09d1', 's09d2', 's09d3', 's09d4', 's10d1', 's10d2', 's10d3', 's10d4']

        # def read_sequence(sequence_path):
        #    sequence = []
        #    for j in os.listdir(sequence_path):
        #        sequence.append(read_json(j))
        #    return sequence

        print('test set folder should be selected from 0 ~ 10')
        print('selected test folder {} includes:'.format(test_set_folder), cross_set[test_set_folder])

        train_set = {}
        test_set = []
        for i in range(len(cross_set)):
            if i == test_set_folder:
                test_set += cross_set[i]
            else:
                train_set[i] = cross_set[i]

        train = {}
        test = {}
        pose_seq_prec = '001'
        pose_class_prec = '1'
        pose_pers_prec = 's01d1'
        sequence = []

        for i in range(0, 10):
            train[i] = {}
            for n in range(1, 7):
                train[i][n] = []
                if i == 0:
                    test[n] = []

        for pose_path in self.pose_paths:
            pose_pers = pose_path.split('/')[-4]
            pose_class = pose_path.split('/')[-3]
            pose_seq = pose_path.split('/')[-2]
            pose = self.read_json(pose_path)
            if pose_seq == pose_seq_prec and pose_class == pose_class_prec:
                if pose:
                    sequence.append(pose)
            else:

                index = -1
                for idx in range(0, len(train_set)):
                    if pose_pers_prec in train_set[idx]:
                        index = idx

                if index != -1:
                    train[index][int(pose_class_prec)].append(sequence)
                    pose_pers_prec = pose_pers
                    pose_seq_prec = pose_seq
                    pose_class_prec = pose_class
                    if pose:
                        sequence = [pose]
                    else:
                        sequence = []

                else:
                    test[int(pose_class_prec)].append(sequence)
                    pose_pers_prec = pose_pers
                    pose_seq_prec = pose_seq
                    pose_class_prec = pose_class
                    if pose:
                        sequence = [pose]
                    else:
                        sequence = []

        return train, test


#Create Dataset Folder
def create_dataset():
    list_class = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    path_dst = '/delorean/aiv/dataset/'
    #modify path server
    path_src = '/delorean/aiv/openpose/output/copy/Jsons/'
    pose_paths = glob.glob(os.path.join(path_src, '*', '*', '*.json'))
    pose_paths.sort()
    sequences_file = open('00sequences.txt', 'r')
    sequences = sequences_file.readlines()

    if not os.path.isdir(path_dst):
        os.mkdir(path_dst)
    print("Creating dataset in:", path_dst)

    for pose_path in pose_paths:

        # extracting important infos from each json to divide them properly

        pose_class = (pose_path.split('/')[-2]).split('_')[-3]
        pose_pers = ((pose_path.split('/')[-2]).split('_')[-4]).split('person')[-1]
        pose_scenario = (pose_path.split('/')[-2]).split('_')[-2]
        json_name = pose_path.split('/')[-1]
        index_class = list_class.index(pose_class) + 1

        json_frame = json_name.split("_")[-2].lstrip("0")
        if json_frame == '':
           json_frame = 1
        else:
           json_frame = int(json_frame) + 1

        # divide jsons into different persons, scenarios and classes
        
        if not os.path.isdir(path_dst + 's' + str(pose_pers) + str(pose_scenario)):
            os.mkdir(path_dst + 's' + str(pose_pers) + str(pose_scenario))
        if not os.path.isdir(path_dst + 's' + str(pose_pers) + str(pose_scenario)  + '/' + str(index_class)):
            os.mkdir(path_dst + 's' + str(pose_pers) + str(pose_scenario)  + '/' + str(index_class))        

        path = path_dst + 's' + str(pose_pers) + str(pose_scenario)  + '/' + str(index_class) + '/'
        
        # divide jsons into sequences, according to sequences file given from repository
        
        match = 'person' + pose_pers + '_' + pose_class + '_' + pose_scenario
        frames_interval = 'plchold'

        for idx in range(0, len(sequences)):
            if match in sequences[idx].split('frames')[0]:
                frames_interval = sequences[idx].split('frames')[1]
                break
        
        print("Json file:", json_name)
        print("Sequences:", frames_interval.replace(" ", ''))
        
        l1, h1 = frames_interval.replace(" ", '').split(",")[0].split("-")
        l2, h2 = frames_interval.replace(" ", '').split(",")[1].split("-")
        l3, h3 = frames_interval.replace(" ", '').split(",")[2].split("-")
        try:
            l4, h4 = frames_interval.replace(" ", '').split(",")[3].split("-")
        except IndexError:
            l4, h4 = [-1, -1]

        # create sequences folder and move json into it, if written into sequences txt file

        move = False

        if int(l1) <= json_frame <= int(h1):
            if not os.path.isdir(path + '001'):
                os.mkdir(path + '001')
            path = path + '001/'
            move = True
        elif int(l2) <= json_frame <= int(h2):
            if not os.path.isdir(path + '002'):
                os.mkdir(path + '002')
            path = path + '002/'
            move = True
        elif int(l3) <= json_frame <= int(h3):
            if not os.path.isdir(path + '003'):
                os.mkdir(path + '003')
            path = path + '003/'
            move = True
        elif int(l4) <= json_frame <= int(h4):
            if not os.path.isdir(path + '004'):
                os.mkdir(path + '004')
            path = path + '004/'
            move = True

        if move:
        
            path = path + json_name

            print("Src_path:", pose_path)
            print("Dst_path:", path)
            print("")
 
            os.rename(pose_path, path)
    
    sequences_file.close()


# Rescale to be 16 frames
def zoom(p):
    l = p.shape[0]
    p_new = np.empty([16, 15, 2])
    for m in range(0, 15):
        for n in range(0, 2):
            p_new[:, m, n] = inter.zoom(p[:, m, n], 16/l)[:16]
    return p_new


# normalization functions of joint values

def normalize_by_joints_distance(tset):
    norm_value = 0
    for n, l in enumerate(tset):
        if np.sum(l) != 0 and not math.isnan(np.sum(l)):
            norm_value = np.linalg.norm(np.subtract(l[2:4], l[4:6])) + np.linalg.norm(np.subtract(l[2:4], l[8:10]))
            l = np.dot(l, norm_value)
            sum_value = np.sum(l)
            l = np.divide(l, sum_value)
            if np.sum(l) != 0 and not math.isnan(np.sum(l)):
                tset[n] = list(l)
    return tset


def normalize_by_center(tset):
    for n, l in enumerate(tset):
        if np.sum(l) != 0:
            center_x = l[2]
            center_y = l[3]
            tset[n][0:len(tset[n]):2] = np.subtract(tset[n][0:len(tset[n]):2], center_x)
            tset[n][1:len(tset[n]):2] = np.subtract(tset[n][1:len(tset[n]):2], center_y)
            tset[n] = list(tset[n])
    return tset


def normalize_by_cosine(tset):
    train_fin = []
    for i, j in enumerate(tset):
        art_sup_dx_1 = [j[8:10], j[10:12]]
        art_sup_sx_1 = [j[2:4], j[4:6]]
        art_inf_dx_1 = [j[20:22], j[22:24]]
        art_inf_sx_1 = [j[14:16], j[16:18]]
        art_sup_dx_2 = [j[10:12], j[12:14]]
        art_sup_sx_2 = [j[4:6], j[6:8]]
        art_inf_dx_2 = [j[22:24], j[24:26]]
        art_inf_sx_2 = [j[16:18], j[18:20]]
        vect_cos = [art_sup_dx_1, art_sup_dx_2, art_sup_sx_1, art_sup_sx_2, art_inf_dx_1, art_inf_dx_2, art_inf_sx_1, art_inf_sx_2]
        train_cos = []
        for l, m in enumerate(vect_cos):
            a = (np.dot(m[0], m[1]))
            b = (np.linalg.norm(m[0]) * np.linalg.norm(m[1]))
            if math.isnan(np.divide(a, b)):
                train_cos.append(0.0)
            else:
                train_cos.append(np.divide(a, b))
        train_fin.append(train_cos)
    return train_fin
