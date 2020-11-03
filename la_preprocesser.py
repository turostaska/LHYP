import math
import os
import pickle
import sys

import pydicom as dicom
import numpy as np
import cv2
import matplotlib.pyplot as plt

DESC_2CH = 'B-TFE_BH 2CH'
DESC_4CH = 'B-TFE_BH 4CH'
DESC_LVOT = 'B-TFE_BH LVOT'

DESC_movie = 'B-TFE_BH_LA movie'

ref_normals = {DESC_2CH: None, DESC_4CH: None, DESC_LVOT: None}
ref_datasets = {DESC_2CH: None, DESC_4CH: None, DESC_LVOT: None}


def to_choose(dataset):
    inst_num = dataset.InstanceNumber % 25
    if inst_num == 0:
        inst_num = 25
    return 9 <= inst_num <= 25 and inst_num % 2 == 1


def dist_of(vec1, vec2):
    return math.hypot(math.hypot(vec1[0] - vec2[0], vec1[1] - vec2[1]), vec1[2] - vec2[2])


def get_desc(dataset):
    if not all(value is not None for value in ref_normals.values()):
        raise Exception('Reference vectors haven\'t been properly initialised yet.')

    desc = None
    dist = None

    normal = get_normal(dataset)

    for key in ref_normals.keys():
        # In case the normal is inverted
        current_dist = min(dist_of(normal, ref_normals[key]), dist_of(-normal, ref_normals[key]))
        if dist is None or current_dist < dist:
            dist = current_dist
            desc = key

    try:
        coded_desc = dataset.SeriesDescription
        if coded_desc != DESC_movie and coded_desc != desc:
            raise Exception('Type mismatch for dataset {}'.format(dataset))
    except KeyError:
        pass

    return desc


def get_normal(dataset):
    return np.cross(dataset.ImageOrientationPatient[3:], dataset.ImageOrientationPatient[:3])


def get_ill(la_folder):
    with open(os.path.join(la_folder, '..', 'meta.txt'), 'r') as meta:
        content = meta.readline().rstrip()
        return content != 'Pathology: Normal'


def get_wcs_matrix(dataset):
    """
    Gives the matrix which transforms the dataset's image to world coordinate system
    Source: https://nipy.org/nibabel/dicom/dicom_orientation.html
    """
    X_x = dataset.ImageOrientationPatient[3]
    X_y = dataset.ImageOrientationPatient[4]
    X_z = dataset.ImageOrientationPatient[5]
    Y_x = dataset.ImageOrientationPatient[0]
    Y_y = dataset.ImageOrientationPatient[1]
    Y_z = dataset.ImageOrientationPatient[2]

    d_i = dataset.PixelSpacing[1]
    d_j = dataset.PixelSpacing[0]

    S_x = dataset.ImagePositionPatient[0]
    S_y = dataset.ImagePositionPatient[1]
    S_z = dataset.ImagePositionPatient[2]

    n = get_normal(dataset)

    M = np.array([[X_x * d_i, X_y * d_i, X_z * d_i, 0],
                  [Y_x * d_j, Y_y * d_j, Y_z * d_j, 0],
                  [n[0] * d_i, n[1] * d_i, n[2] * d_i, 0],
                  [S_x, S_y, S_z, 1]])
    return np.transpose(M)


def apply_affine(dataset):
    """
    Applies affine transformation to the dataset's image so that the patient's position will match the position of the
    reference patient
    @return: the transformed pixel array
    """
    desc = get_desc(dataset)
    affine = np.matmul(np.linalg.pinv(get_wcs_matrix(ref_datasets[desc])), get_wcs_matrix(dataset))

    img = dataset.pixel_array
    img = (img / img.max() * 255).astype(np.uint8).transpose()

    M = np.array([[affine[0, 0], affine[0, 1], affine[0, 3]],
                  [affine[1, 0], affine[1, 1], affine[1, 3]]])
    rows, cols = img.shape

    tr = cv2.warpAffine(img, M, (cols, rows)).transpose()
    tr = np.clip(tr, np.percentile(tr, 5), np.percentile(tr, 95))
    tr = (tr / tr.max() * 255).astype(np.uint8)
    return tr


class ProcessedData:
    def __init__(self, folder):
        self.imgs_2ch = []
        self.imgs_4ch = []
        self.imgs_lvot = []
        self.is_ill = get_ill(folder)
        self.__choose_imgs(folder)
        self.__sort_imgs()

    def __choose_imgs(self, la_folder):
        la_files = sorted(os.listdir(la_folder))
        for la_file in la_files:
            if is_dicom(la_file):
                try:
                    dataset = dicom.dcmread(os.path.join(la_folder, la_file))
                    if to_choose(dataset):
                        self.__add_img(dataset)

                except KeyError as ke:
                    print(ke)

    def __add_img(self, dataset):
        try:
            desc = get_desc(dataset)
            img = apply_affine(dataset)
            inst_num = dataset.InstanceNumber % 25 if dataset.InstanceNumber % 25 != 0 else 25
            if desc == DESC_2CH:
                if not [item for item in self.imgs_2ch if item[1] == inst_num]:
                    self.imgs_2ch.append((img, inst_num))
            elif desc == DESC_4CH:
                if not [item for item in self.imgs_4ch if item[1] == inst_num]:
                    self.imgs_4ch.append((img, inst_num))
            elif desc == DESC_LVOT:
                if not [item for item in self.imgs_lvot if item[1] == inst_num]:
                    self.imgs_lvot.append((img, inst_num))
        except (ValueError, AttributeError) as e:
            print(e)

    def __sort_imgs(self):
        comp = lambda x: x[1]
        self.imgs_2ch.sort(key=comp)
        self.imgs_4ch.sort(key=comp)
        self.imgs_lvot.sort(key=comp)
        pass


def is_dicom(file_name):
    return file_name.lower().endswith('.dcm')


def is_relevant(la_folder):
    with open(os.path.join(la_folder, '..', 'meta.txt'), 'r') as meta:
        content = meta.readline().rstrip().replace('Pathology: ', '').lower()
        return content != 'u18_m' and content != 'u18_f' and content != 'adult_f_sport' and content != 'adult_m_sport'


def assign_ref_vectors(src):
    folders = sorted(os.listdir(src))

    for folder in folders:
        la_folder = os.path.join(src, folder, 'la')
        la_files = sorted(os.listdir(la_folder))

        for file in la_files:
            if is_dicom(file):
                try:
                    # Assign value to ref_vec if it's null and dataset has a series description
                    dataset = dicom.dcmread(os.path.join(la_folder, file))
                    desc = dataset.SeriesDescription
                    if desc in ref_normals:
                        if ref_normals[desc] is None:
                            ref_normals[desc] = get_normal(dataset)
                            ref_datasets[desc] = dataset

                except Exception as e:
                    raise e

                # If there is one reference vector for all types of series descriptions, break the loop
                if all(value is not None for value in ref_normals.values()):
                    return


def save_state(folder):
    with open('state', 'w') as state:
        state.write(folder)


def load_state():
    try:
        with open('state', 'r') as state:
            return state.readline().rstrip()
    except:
        return None


def process_la(src, dest):
    last_saved = load_state()

    folders = sorted(os.listdir(src))
    for folder in folders:
        la_folder = os.path.join(src, folder, 'la')
        if not is_relevant(la_folder) or not os.listdir(la_folder):
            continue

        if last_saved is not None:
            if folder == last_saved:
                last_saved = None
            continue

        proc_data = ProcessedData(la_folder)

        # for img in proc_data.imgs_2ch:
        #     f, axarr = plt.subplots(1, 1)
        #     axarr.imshow(img[0])
        #     plt.show()

        with open(os.path.join(dest, folder), 'wb') as f:
            pickle.dump(proc_data, f)

        save_state(folder)


def main(src, dest):
    assign_ref_vectors(src)
    process_la(src, dest)


if __name__ == "__main__":
    source = '/home/albert/Documents/LHYP-master/hypertrophy/' if len(sys.argv) < 2 else sys.argv[1]
    destination = './output' if len(sys.argv) < 3 else sys.argv[2]
    main(source, destination)
