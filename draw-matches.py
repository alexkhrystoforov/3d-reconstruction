import cv2
import numpy as np


class Matcher:

    def __init__(self, algorithm, img0, img1):
        self.algorithm = algorithm
        self.kp1, self.des1 = algorithm.detectAndCompute(img0, None)
        self.kp2, self.des2 = algorithm.detectAndCompute(img1, None)

        self.norm_hamming = cv2.NORM_HAMMING2

    def BF_matcher(self, mode):

        if mode == 'SIFT' or mode == 'KAZE':
            bf = cv2.BFMatcher(crossCheck=True)

        if mode == 'ORB':
            bf = cv2.BFMatcher(self.norm_hamming, crossCheck=True)

        matches = bf.match(self.des2, self.des1)
        matches = sorted(matches, key=lambda x: x.distance)
        best_matches = find_best_matches(matches)

        return best_matches


def find_best_matches(matches):
    """
    Filter matches by distance
    Args:
         matches: list
    Returns:
        best_matches: list
    """
    best_matches = []
    for m in matches:
        if m.distance < 200:  # Матчинг - мы сравниваем 2 вектора. сортируем матчи по растоянию,
            # тут можно делать приемлимое расстояние больше или меньше
            best_matches.append(m)
    return best_matches


def draw_matches(imgL, imgR, mode='SIFT'):
    MIN_MATCH_COUNT = 100  # минимальное число заматченных точек, чтоб отрисовало

    if mode == 'ORB':
        algorithm = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)

    if mode == 'SIFT':
        algorithm = cv2.SIFT_create()

    if mode == 'KAZE':
        algorithm = cv2.KAZE_create()


    matcher = Matcher(algorithm, imgL, imgR)
    bf_matches = matcher.BF_matcher(mode)

    print(f'количество заматченный точек после фильтра: {mode}', len(bf_matches))

    if len(bf_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([matcher.kp2[m.queryIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([matcher.kp1[m.trainIdx].pt for m in bf_matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        draw_params = dict(outImg=None,
                           # matchColor=(0, 255, 0),
                           # matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        result_mathing_bf = cv2.drawMatches(imgR, matcher.kp2, imgL, matcher.kp1, bf_matches, **draw_params)

        cv2.namedWindow('draw_matches', cv2.WINDOW_NORMAL)
        cv2.imshow('draw_matches', result_mathing_bf)
        cv2.waitKey(0)

        if mode == 'KAZE':
            cv2.imwrite('res_kaze.png', result_mathing_bf)
        if mode == 'SIFT':
            cv2.imwrite('res_sift.png', result_mathing_bf)
        if mode == 'ORB':
            cv2.imwrite('res_orb.png', result_mathing_bf)


imgL = cv2.imread('sets/ringL.png')
imgR = cv2.imread('sets/ringR.png')

draw_matches(imgL, imgR, mode='SIFT')
# draw_matches(imgL, imgR, mode='KAZE')
# draw_matches(imgL, imgR, mode='ORB')
