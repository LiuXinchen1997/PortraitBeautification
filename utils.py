import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt


class FaceDetector:
    def __init__(self, detector_path='./data/shape_predictor_68_face_landmarks.dat'):
        self.detector_path = detector_path

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.detector_path)

    def read_image(self, image_path):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        return img

    def get_bgr(self, image_path):
        return self.read_image(image_path)

    def get_hsv(self, image_path):
        img_bgr = self.get_bgr(image_path)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    def get_face_rect_and_landmarks(self, image_path):
        img_bgr = self.get_bgr(image_path)
        img_hsv = self.get_hsv(image_path)

        rects = self.detector(img_bgr, 1)
        landmarks = np.array([[p.x, p.y] for p in self.predictor(img_bgr, rects[0]).parts()])

        return rects[0], landmarks


class Organ:
    def __init__(self, img_bgr, landmarks, name):
        self.original_img_bgr = img_bgr.copy()
        self.img_bgr = img_bgr
        self.landmarks = landmarks
        self.name = name

        self.top, self.bottom, self.left, self.right = self._get_rect()
        self.shape = (int(self.bottom - self.top), int(self.right - self.left))
        self.size = self.shape[0] * self.shape[1] * 3
        self.move = int(np.sqrt(self.size / 3) / 20.)
        # self.move = 0
        self.ksize = self._get_ksize()
        self.patch_bgr = self._get_patch(self.img_bgr)
        self.patch_mask = self._get_re_mask()

    def _get_ksize(self, rate=15):
        # ksize: kernel size
        size = max([int(np.sqrt(self.size / 3) / rate), 1])
        size = (size if size % 2 == 1 else size + 1)
        return (size, size)

    def _get_rect(self):
        ys, xs = self.landmarks[:, 1], self.landmarks[:, 0]
        return np.min(ys), np.max(ys), np.min(xs), np.max(xs)

    def _get_patch(self, img):
        shape = img.shape
        patch = img[np.max([self.top - self.move, 0]):np.min([self.bottom + self.move, shape[0]]),
                np.max([self.left - self.move, 0]):np.min([self.right + self.move, shape[1]])]
        return patch

    def _draw_convex_hull(self, img, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(img, points, color=color)

    def _get_re_mask(self, ksize=None):
        if ksize is None:
            ksize = self.ksize
        landmarks_re = self.landmarks.copy()
        landmarks_re[:, 1] -= np.max([self.top - self.move, 0])
        landmarks_re[:, 0] -= np.max([self.left - self.move, 0])
        mask = np.zeros(self.patch_bgr.shape[:2], dtype=np.float64)

        self._draw_convex_hull(mask, landmarks_re, color=1)
        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
        mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        return cv2.GaussianBlur(mask, ksize, 0)[:]

    def get_abs_mask(self, ksize=None):
        if ksize is None:
            ksize = self.ksize
        mask = np.zeros(self.img_bgr.shape, dtype=np.float64)
        patch = self._get_patch(mask)
        patch[:] = self.patch_mask[:]

        return mask

    def whitening(self, rate=0.15):
        img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        patch_hsv = self._get_patch(img_hsv)
        patch_hsv[:, :, -1] = np.minimum(
            patch_hsv[:, :, -1] + patch_hsv[:, :, -1] * self.patch_mask[:, :, -1] * rate, 255).astype('uint8')
        self.img_bgr[:] = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)[:]

    def brightening(self, rate=0.15):
        original_img_bgr = cv2.cvtColor(self.original_img_bgr, cv2.COLOR_BGR2HSV)
        img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)

        original_patch_hsv = self._get_patch(original_img_bgr)
        patch_hsv = self._get_patch(img_hsv)
        patch_hsv[:, :, 1] = np.minimum(
            original_patch_hsv[:, :, 1] + original_patch_hsv[:, :, 1] * self.patch_mask[:, :, 1] * rate, 255).astype('uint8')
        self.img_bgr[:] = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)[:]

    def largeeye(self, rate=0.15):
        print(self.img_bgr.shape)  # height * width * channel
        pass

    def slimface(self, rate=0.15):
        print(self.img_bgr.shape)  # height * width * channel
        pass

    def test(self, rate=0.15):
        self.patch_bgr[:, :, 0][self.patch_bgr[:, :, 0] * self.patch_mask[:, :, 0] > 0] = 0
        self.patch_bgr[:, :, 1][self.patch_bgr[:, :, 1] * self.patch_mask[:, :, 1] > 0] = 255
        self.patch_bgr[:, :, 2][self.patch_bgr[:, :, 2] * self.patch_mask[:, :, 2] > 0] = 0


class Forehead(Organ):
    def __init__(self, img_bgr, landmarks, mask_organs, name):
        self.mask_organs = mask_organs
        super(Forehead, self).__init__(img_bgr, landmarks, name)

    def _get_re_mask(self, ksize=None):
        if ksize is None:
            ksize = self.ksize
        landmarks_re = self.landmarks.copy()
        landmarks_re[:, 1] -= np.max([self.top - self.move, 0])
        landmarks_re[:, 0] -= np.max([self.left - self.move, 0])
        mask = np.zeros(self.patch_bgr.shape[:2], dtype=np.float64)

        self._draw_convex_hull(mask, landmarks_re, color=1)
        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
        mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        patch_organs = self._get_patch(self.mask_organs)
        mask = cv2.GaussianBlur(mask, ksize, 0)[:]
        mask[patch_organs > 0] = (1 - patch_organs[patch_organs > 0])
        return mask


class Face(Organ):
    def __init__(self, img_bgr, landmarks):
        self.original_img_bgr = img_bgr.copy()
        self.organs_name = ['jaw', 'mouth', 'nose', 'left eye', 'right eye', 'left brow', 'right brow']
        self.organs_points = [list(range(0, 17)), list(range(48, 61)), list(range(27, 35)), list(range(42, 48)),
                              list(range(36, 42)), list(range(22, 27)), list(range(17, 22))]
        self.organs = {name: Organ(img_bgr, landmarks[points], name) for name, points in
                       zip(self.organs_name, self.organs_points)}

        # 获得额头坐标，实例化额头
        mask_nose = self.organs['nose'].get_abs_mask()
        mask_organs = (self.organs['mouth'].get_abs_mask() + mask_nose + self.organs['left eye'].get_abs_mask() +
                       self.organs['right eye'].get_abs_mask() + self.organs['left brow'].get_abs_mask() +
                       self.organs['right brow'].get_abs_mask())
        forehead_landmarks = self._get_forehead_landmark(img_bgr, landmarks, mask_organs, mask_nose)

        self.organs['forehead'] = Forehead(img_bgr, forehead_landmarks, mask_organs, 'forehead')
        self.mask_organs = mask_organs.copy()
        self.mask_organs += self.organs['forehead'].get_abs_mask()
        mask_organs += self.organs['forehead'].get_abs_mask()

        # 人脸的完整标记点
        self.face_points = np.concatenate([landmarks, forehead_landmarks])
        super(Face, self).__init__(img_bgr, self.face_points, 'face')

        mask_face = self.get_abs_mask() - mask_organs
        self.patch_mask = self._get_patch(mask_face)

    def _get_forehead_landmark(self, img_bgr, face_landmark, mask_organs, mask_nose):
        # 绘制额头
        radius = (np.linalg.norm(face_landmark[0] - face_landmark[16]) / 2).astype('int32')
        abs_center = tuple(((face_landmark[0] + face_landmark[16]) / 2).astype('int32'))

        angle = np.degrees(np.arctan((lambda l: l[1] / l[0])(face_landmark[16] - face_landmark[0]))).astype('int32')
        mask = np.zeros(mask_organs.shape[:2], dtype=np.float64)
        cv2.ellipse(mask, abs_center, (radius, radius), angle, 180, 360, 1, -1)
        # 剔除与五官重合部分
        mask[mask_organs[:, :, 0] > 0] = 0
        # 根据鼻子的肤色判断真正的额头面积，提出头发等其他部件
        index_bool = []
        for ch in range(3):
            mean, std = np.mean(img_bgr[:, :, ch][mask_nose[:, :, ch] > 0]), np.std(
                img_bgr[:, :, ch][mask_nose[:, :, ch] > 0])
            up, down = mean + 0.5 * std, mean - 0.5 * std
            index_bool.append((img_bgr[:, :, ch] < down) | (img_bgr[:, :, ch] > up))
        index_zero = ((mask > 0) & index_bool[0] & index_bool[1] & index_bool[2])
        mask[index_zero] = 0
        index_abs = np.array(np.where(mask > 0)[::-1]).transpose()
        landmarks = cv2.convexHull(index_abs).squeeze()
        return landmarks

    def whitening(self, rate=0.15):
        original_img_hsv = cv2.cvtColor(self.original_img_bgr, cv2.COLOR_BGR2HSV)
        img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)

        full_face_mask = self.get_abs_mask() + self.mask_organs
        img_hsv[:, :, -1] = np.minimum(
            original_img_hsv[:, :, -1] + original_img_hsv[:, :, -1] * full_face_mask[:, :, -1] * rate, 255).astype('uint8')
        self.img_bgr[:] = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)[:]


if __name__ == '__main__':
    image_path = './images/1.jpg'

    detector = FaceDetector()
    bgr = detector.get_bgr(image_path)
    _, landmarks = detector.get_face_rect_and_landmarks(image_path)

    # 显示掩膜
    face = Face(bgr, landmarks)
    print(face.mask_organs.shape)
    print(face.mask_organs)

    #cv2.imshow('test', face.organs['forehead'].get_abs_mask())
    #cv2.imshow('test', face.get_abs_mask())  # 除去额头、眼睛、眉毛、鼻子、嘴的其他区域
    #cv2.imshow('test', face.mask_organs)
    cv2.imshow('test', face.organs['left brow'].get_abs_mask())
    cv2.waitKey(0)
    cv2.destroyWindow('test')

