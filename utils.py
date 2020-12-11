import cv2


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
        if m.distance < 300:
            best_matches.append(m)

    return best_matches


class Matcher:

    def __init__(self, orb, img0, img1):
        self.kp1, self.des1 = orb.detectAndCompute(img0, None)
        self.kp2, self.des2 = orb.detectAndCompute(img1, None)
        self.norm_hamming = cv2.NORM_HAMMING

    def BF_matcher(self):
        bf = cv2.BFMatcher(crossCheck=True)
        # bf = cv2.BFMatcher(self.norm_hamming, crossCheck=True)
        matches = bf.match(self.des1, self.des2)
        matches = sorted(matches, key=lambda x: x.distance)
        best_matches = find_best_matches(matches)

        return best_matches