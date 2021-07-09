from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json, read_json


def _pluck(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid))
                else:
                    ret.append((fname, pid, camid))
    return ret


def _pluck_gallery(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            if len(cam_images[:-1])==0:
                for fname in cam_images:
                    name = osp.splitext(fname)[0]
                    x, y, _ = map(int, name.split('_'))
                    assert pid == x and camid == y
                    if relabel:
                        ret.append((fname, index, camid))
                    else:
                        ret.append((fname, pid, camid))
            else:
                for fname in cam_images[:-1]:
                    name = osp.splitext(fname)[0]
                    x, y, _ = map(int, name.split('_'))
                    assert pid == x and camid == y
                    if relabel:
                        ret.append((fname, index, camid))
                    else:
                        ret.append((fname, pid, camid))
    return ret


def _pluck_query(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images[-1:]:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid))
                else:
                    ret.append((fname, pid, camid))
    return ret


class CUHK03(object):
    def __init__(self, root, split_id=0, num_val=100, verbose=True):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_pids, self.num_val_pids, self.num_trainval_pids = 0, 0, 0
        self.min_cam, self.max_cam = 999, 0

        self.prepare()

        self.load(num_val, verbose=verbose)

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def get_num_cams(self, data):
        cams = []
        for _, _, camid in data:
            cams += [camid]
            self.min_cam = min(self.min_cam, camid)
            self.max_cam = max(self.max_cam, camid)
        cams = set(cams)
        num_cams = len(cams)
        return num_cams

    def load(self, num_val=0.3, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        # Randomly split train / val
        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        num = len(trainval_pids)
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        self.train = _pluck(identities, train_pids, relabel=True)
        self.val = _pluck(identities, val_pids, relabel=True)
        self.trainval = _pluck(identities, trainval_pids, relabel=True)
        self.query = _pluck_query(identities, self.split['query'])
        self.gallery = _pluck_gallery(identities, self.split['gallery'])
        self.num_train_pids = len(train_pids)
        self.num_val_pids = len(val_pids)
        self.num_trainval_pids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images | # cameras")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d} | {:5d}"
                  .format(self.num_train_pids, len(self.train), self.get_num_cams(self.train)))
            print("  val      | {:5d} | {:8d} | {:5d}"
                  .format(self.num_val_pids, len(self.val), self.get_num_cams(self.val)))
            print("  trainval | {:5d} | {:8d} | {:5d}"
                  .format(self.num_trainval_pids, len(self.trainval), self.get_num_cams(self.trainval)))
            print("  query    | {:5d} | {:8d} | {:5d}"
                  .format(len(self.split['query']), len(self.query), self.get_num_cams(self.query)))
            print("  gallery  | {:5d} | {:8d} | {:5d}"
                  .format(len(self.split['gallery']), len(self.gallery), self.get_num_cams(self.gallery)))
            print('cam id min is {}, max is {}'. format(self.min_cam, self.max_cam))

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))

    def prepare(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import h5py
        from scipy.misc import imsave

        exdir = osp.join(self.root, 'cuhk03_release')

        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)
        matdata = h5py.File(osp.join(exdir, 'cuhk-03.mat'), 'r')

        def deref(ref):
            return matdata[ref][:].T

        def dump_(refs, pid, cam, fnames):
            for ref in refs:
                img = deref(ref)
                if img.size == 0 or img.ndim < 2: break
                fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, cam, len(fnames))
                imsave(osp.join(images_dir, fname), img)
                fnames.append(fname)

        identities = []
        for labeled, detected in zip(
                matdata['labeled'][0], matdata['detected'][0]):
            labeled, detected = deref(labeled), deref(detected)
            assert labeled.shape == detected.shape
            for i in range(labeled.shape[0]):
                pid = len(identities)
                images = [[], []]
                dump_(labeled[i, :5], pid, 0, images[0])
                dump_(detected[i, :5], pid, 0, images[0])
                dump_(labeled[i, 5:], pid, 1, images[1])
                dump_(detected[i, 5:], pid, 1, images[1])
                identities.append(images)

        # Save meta information into a json file
        meta = {'name': 'cuhk03', 'shot': 'multiple', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save training and test splits
        splits = []
        view_counts = [deref(ref).shape[0] for ref in matdata['labeled'][0]]
        vid_offsets = np.r_[0, np.cumsum(view_counts)]
        for ref in matdata['testsets'][0]:
            test_info = deref(ref).astype(np.int32)
            test_pids = sorted(
                [int(vid_offsets[i-1] + j - 1) for i, j in test_info])
            trainval_pids = list(set(range(vid_offsets[-1])) - set(test_pids))
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
        print('prepared', osp.join(self.root, 'meta.json'))
        print('prepared', osp.join(self.root, 'splits.json'))

