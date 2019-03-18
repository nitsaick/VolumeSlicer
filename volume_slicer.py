import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin


class VolumeSlicer:
    TaitBryanAngle = 0
    AxisAngle = 1

    def __call__(self, vol, dst_norm_vec, dst_center, size, crop=False, method=TaitBryanAngle):
        if isinstance(size, (tuple, list)):
            img_x, img_y = size
        else:
            img_x = img_y = size

        src_center = np.array([img_x / 2, img_y / 2, 0.0])
        src_coor = np.array([(i, j, 0) for i in range(img_x) for j in range(img_y)], dtype=np.float)
        src_coor = src_coor.reshape((img_x, img_y, 3))
        src_norm_vec = np.array([0.0, 0.0, 1.0])

        if method == self.TaitBryanAngle:
            r_mat = self._get_rotation_mat(src_norm_vec, dst_norm_vec)
        elif method == self.AxisAngle:
            r_mat = self._get_rodrigues_rotation_mat(src_norm_vec, dst_norm_vec)
        else:
            raise ValueError('Unknown method {}. Allow method: TaitBryanAngle, AxisAngle'.format(method))

        t_mat = self._get_translation_mat(src_center, dst_center, r_mat)
        dst_coor = self._transformation(src_coor, r_mat, t_mat)

        img = self._slice_img_from_vol(vol, dst_coor)

        if crop:
            img = self._remove_black_region(img)

        return img

    @staticmethod
    def _get_two_vec_angle(src, dst):
        c = np.dot(src, dst) / np.linalg.norm(src) / np.linalg.norm(dst)
        angle = np.arccos(np.clip(c, -1, 1))
        cross = np.cross(src, dst)
        if cross < 0:
            angle *= -1
        if np.isnan(angle):
            angle = 0.0
        return angle

    @staticmethod
    def _get_rotation_mat(src_norm_vec, dst_norm_vec):
        ax = VolumeSlicer._get_two_vec_angle([src_norm_vec[1], src_norm_vec[2]], [dst_norm_vec[1], dst_norm_vec[2]])
        ay = VolumeSlicer._get_two_vec_angle([src_norm_vec[2], src_norm_vec[0]], [dst_norm_vec[2], dst_norm_vec[0]])
        az = VolumeSlicer._get_two_vec_angle([src_norm_vec[0], src_norm_vec[1]], [dst_norm_vec[0], dst_norm_vec[1]])

        rx = np.array([[1.0, 0.0, 0.0], [0.0, cos(ax), -sin(ax)], [0.0, sin(ax), cos(ax)]])
        ry = np.array([[cos(ay), 0.0, sin(ay)], [0.0, 1.0, 0.0], [-sin(ay), 0.0, cos(ay)]])
        rz = np.array([[cos(az), -sin(az), 0.0], [sin(az), cos(az), 0.0], [0.0, 0.0, 1.0]])

        r_mat = np.dot(rx, np.dot(ry, rz))
        return r_mat

    @staticmethod
    def _get_rodrigues_rotation_mat(src_norm_vec, dst_norm_vec):
        axis = np.cross(src_norm_vec, dst_norm_vec)
        axis = axis / np.linalg.norm(axis)
        axis = [0.0 if np.isnan(i) else i for i in axis]

        angle = np.arccos(
            np.dot(src_norm_vec, dst_norm_vec) / np.linalg.norm(src_norm_vec) / np.linalg.norm(dst_norm_vec))

        r_mat = np.zeros([3, 3])
        r_mat[0, 0] = cos(angle) + axis[0] * axis[0] * (1 - cos(angle))
        r_mat[0, 1] = axis[0] * axis[1] * (1 - cos(angle) - axis[2] * sin(angle))
        r_mat[0, 2] = axis[1] * sin(angle) + axis[0] * axis[2] * (1 - cos(angle))
        r_mat[1, 0] = axis[2] * sin(angle) + axis[0] * axis[1] * (1 - cos(angle))
        r_mat[1, 1] = cos(angle) + axis[1] * axis[1] * (1 - cos(angle))
        r_mat[1, 2] = -axis[0] * sin(angle) + axis[1] * axis[2] * (1 - cos(angle))
        r_mat[2, 0] = -axis[1] * sin(angle) + axis[0] * axis[2] * (1 - cos(angle))
        r_mat[2, 1] = axis[0] * sin(angle) + axis[1] * axis[2] * (1 - cos(angle))
        r_mat[2, 2] = cos(angle) + axis[2] * axis[2] * (1 - cos(angle))

        return r_mat

    @staticmethod
    def _get_translation_mat(src_pt, dst_pt, r_mat):
        src_pt = np.dot(r_mat, src_pt)
        t_mat = dst_pt - src_pt
        return t_mat

    @staticmethod
    def _transformation(plane_coor, r_mat, t_mat):
        tran_plane = np.zeros_like(plane_coor)

        for i in range(len(plane_coor)):
            for j in range(len(plane_coor[i])):
                coor = np.dot(r_mat, plane_coor[i, j])
                for k in range(3):
                    coor[k] += t_mat[k]
                tran_plane[i, j] = coor

        return tran_plane

    @staticmethod
    def _slice_img_from_vol(vol, plane_coor):
        img_x, img_y = plane_coor.shape[0], plane_coor.shape[1]
        img = np.zeros((img_x, img_y))
        for i in range(img_x):
            for j in range(img_y):
                coor = np.around(plane_coor[i, j]).astype(np.int)
                if vol.shape[0] > coor[0] >= 0 and vol.shape[1] > coor[1] >= 0 and vol.shape[2] > coor[2] >= 0:
                    img[i, j] = vol[coor[0], coor[1], coor[2]]
                else:
                    img[i, j] = 0

        return img

    @staticmethod
    def _remove_black_region(src):
        v = np.sum(src, axis=0)
        h = np.sum(src, axis=1)
        v_idx = np.where(v > 0)
        h_idx = np.where(h > 0)
        left = v_idx[0][0]
        right = v_idx[0][-1]
        top = h_idx[0][0]
        bottom = h_idx[0][-1]
        dst = src[top:bottom + 1, left:right + 1]

        return dst


if __name__ == '__main__':
    with open('sample/data0_slice_norm_vec.txt') as f:
        slice_norm_vec = [float(vec) for vec in f.readlines()]
    with open('sample/data0_slice_center.txt') as f:
        slice_center = [float(coor) for coor in f.readlines()]
    vol = np.load('sample/data0_volume.npy')
    img_size = (400, 400)

    slicer = VolumeSlicer()
    img = slicer(vol, slice_norm_vec, slice_center, img_size, crop=True, method=VolumeSlicer.TaitBryanAngle)

    plt.figure()
    plt.imshow(img)
    plt.show()
