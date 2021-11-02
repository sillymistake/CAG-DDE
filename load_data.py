import os
import glob
import numpy as np

TAG_CHAR = np.array([202021.25], np.float32)
def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            return np.resize(data, (int(h), int(w), 2))

def load_davis():
    root_dir = './dataset/DAVIS/DAVIS/'
    file_dir = 'ImageSets/480p/'
    train_file = 'train.txt'
    test_file = 'val.txt'

    tra_dir = []
    for line in open(root_dir+file_dir+train_file, 'r'):
        tra_dir.append(line)

    tes_dir = []
    for line in open(root_dir+file_dir+test_file, 'r'):
        tes_dir.append(line)

    # [image, optical, label]
    tra_list = []
    tes_list = []

    for line in tra_dir:
        image_dir, label_dir = line.split()
        flo_dir = image_dir.replace('JPEGImages', 'optical')
        # flo_dir = flo_dir.replace('jpg', 'flo')
        tra_list.append({
            'img': root_dir+image_dir,
            'flo': root_dir+flo_dir,
            'lbl': root_dir+label_dir
            })

    for line in tes_dir:
        image_dir, label_dir = line.split()
        flo_dir = image_dir.replace('JPEGImages', 'optical')
        # flo_dir = flo_dir.replace('jpg', 'flo')
        tes_list.append({
            'img': root_dir+image_dir,
            'flo': root_dir+flo_dir,
            'lbl': root_dir+label_dir
            })

    return tra_list, tes_list

# only for testing
def load_fbms():
    root_dir = './dataset/FBMS/FBMS/'
    file_dir = 'ImageSets/'
    test_file = 'test.txt'

    tes_dir = []
    for line in open(root_dir+file_dir+test_file, 'r'):
        tes_dir.append(line)

    # [image, optical, label]
    tes_list = []

    for line in tes_dir:
        image_dir, label_dir = line.split()
        flo_dir = image_dir.replace('JPEGImages', 'optical')
        # flo_dir = flo_dir.replace('jpg', 'flo')
        tes_list.append({
            'img': root_dir+image_dir,
            'flo': root_dir+flo_dir,
            'lbl': root_dir+label_dir
            })

    return tes_list

# all for testing
def load_segv2():
    root_dir = './dataset/SegV2/SegV2/'
    img_dir = 'Imgs'

    tes_img_name_list = []

    ins_list = os.listdir(os.path.join(root_dir))
    for ins in ins_list:
        img_list = os.listdir(os.path.join(root_dir,ins,img_dir))
        for img in img_list:
            tes_img_name_list.append(os.path.join(root_dir,ins,img_dir,img))

    tes_list = []
    for image_dir in tes_img_name_list:
        flo_dir = image_dir.replace('Imgs', 'optical')
        lal_dir = image_dir.replace('Imgs', 'ground-truth').replace('jpg', 'png')
        # flo_dir = flo_dir.replace('jpg', 'flo')
        tes_list.append({
            'img': image_dir,
            'flo': flo_dir,
            'lbl': lal_dir
            })

    return tes_list

def load_davsod():
    root_dir = './dataset/'
    file_dir = 'DAVSOD/'
    train_file = 'train.txt'
    test_file = 'easy.txt'

    tra_dir = []
    for line in open(root_dir+file_dir+train_file, 'r'):
        tra_dir.append(line.replace('\n',''))

    tes_dir = []
    for line in open(root_dir+file_dir+test_file, 'r'):
        tes_dir.append(line.replace('\n',''))

    # [image, optical, label]
    tra_list = []
    tes_list = []

    for line in tra_dir:
        image_dir, label_dir = line.split(',')
        flo_dir = image_dir.replace('Imgs', 'optical')
        ins = image_dir.split('/')[-3]
        tra_list.append({
            'img': root_dir+image_dir,
            'flo': root_dir+flo_dir,
            'lbl': root_dir+label_dir
            })

    for line in tes_dir:
        image_dir, label_dir = line.split(',')
        flo_dir = image_dir.replace('Imgs', 'optical')
        tes_list.append({
            'img': root_dir+image_dir,
            'flo': root_dir+flo_dir,
            'lbl': root_dir+label_dir
            })

    return tra_list, tes_list

def load_duts():
    root_dir = './dataset/'
    tra_image_dir = 'DUTS-TR/DUTS-TR/DUTS-TR-Image/'
    tra_label_dir = 'DUTS-TR/DUTS-TR/DUTS-TR-Mask/'
    tes_image_dir = 'DUTS-TE/DUTS-TE/DUTS-TE-Image/'
    tes_label_dir = 'DUTS-TE/DUTS-TE/DUTS-TE-Mask/'

    image_ext = '.jpg'
    label_ext = '.png'

    tra_img_name_list = glob.glob(root_dir + tra_image_dir + '*' + image_ext)
    tra_list = []

    for img_path in tra_img_name_list:
        img_name = img_path.split('/')[-1]
        tmp = img_name.split('.')[0:-1]
        imidx = tmp[0]
        for i in range(1, len(tmp)):
            imidx = imidx + '.' + tmp[i]
        tra_list.append({
            'img': img_path,
            'flo': img_path,
            'lbl': root_dir + tra_label_dir + imidx + label_ext
            })

    # test dataset
    tes_img_name_list = glob.glob(root_dir + tes_image_dir + '*' + image_ext)
    tes_list = []

    for img_path in tes_img_name_list:
        img_name = img_path.split('/')[-1]
        tmp = img_name.split('.')[0:-1]
        imidx = tmp[0]
        for i in range(1, len(tmp)):
            imidx = imidx + '.' + tmp[i]
        tes_list.append({
            'img': img_path,
            'flo': img_path, # fake
            'lbl': root_dir + tes_label_dir + imidx + label_ext
            })

    return tra_list, tes_list



if __name__ == '__main__':
    from tqdm import tqdm
    from PIL import Image

    tes_list = load_segv2()
    for i in tqdm(range(len(tes_list))):
        img = Image.open(tes_list[i]['img']).convert('RGB')
        flo = Image.open(tes_list[i]['flo']).convert('RGB')
        lbl = Image.open(tes_list[i]['lbl']).convert('L')
        assert img.size ==  flo.size
