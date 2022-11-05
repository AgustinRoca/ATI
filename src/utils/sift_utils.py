import cv2 as cv

def key_points(filename):
    img = cv.imread(filename)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)

    img = cv.drawKeypoints(gray,kp,img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img

def sift(filename_1: str, filename_2: str, threshold, m = 10, keypoints_1=None, descriptors_1=None):
    img_1 = cv.imread(filename_1)
    img_2 = cv.imread(filename_2)

    img_1 = cv.cvtColor(img_1,cv.COLOR_BGR2GRAY)
    img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    if keypoints_1 == None:
        keypoints_1, descriptors_1 = sift.detectAndCompute(img_1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img_2,None)

    #feature matching
    bf = cv.BFMatcher_create(cv.NORM_L2, crossCheck=False)

    matches = bf.match(descriptors_1, descriptors_2)
    total_matches_count = len(matches)
    valid_matches = []
    for match in matches:
        if match.distance <= threshold:
            valid_matches.append(match)
    valid_matches_count = len(valid_matches)
    valid_matches = sorted(valid_matches, key = lambda x:x.distance)

    max_matches = min(valid_matches_count, m)

    img_3 = cv.drawMatches(img_1, keypoints_1, img_2, keypoints_2, valid_matches[:max_matches], img_2, flags=2)

    return img_3, valid_matches_count/total_matches_count, keypoints_1, descriptors_1


def sift_detector(filename_1: str, filename_2: str, threshold, n, m = 10, keypoints_1=None, descriptors_1=None):
    img_1 = cv.imread(filename_1)
    img_2 = cv.imread(filename_2)

    img_1 = cv.cvtColor(img_1,cv.COLOR_BGR2GRAY)
    img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    if keypoints_1 == None:
        keypoints_1, descriptors_1 = sift.detectAndCompute(img_1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img_2,None)

    #feature matching
    bf = cv.BFMatcher_create(cv.NORM_L2, crossCheck=False)

    matches = bf.match(descriptors_1, descriptors_2)
    total_matches_count = len(matches)
    valid_matches = []
    for match in matches:
        if match.distance <= threshold:
            valid_matches.append(match)
    valid_matches_count = len(valid_matches)
    valid_matches = sorted(valid_matches, key = lambda x:x.distance)

    max_matches = min(valid_matches_count, m)

    p1, p2 = get_rectangle(valid_matches, keypoints_2, img_2.shape[0], img_2.shape[1])

    if valid_matches_count > n:
        img_3 = cv.rectangle(img_2, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 2)
    else: 
        img_3 = img_2
    img_3 = cv.drawMatches(img_1, keypoints_1, img_3, keypoints_2, valid_matches[:max_matches], img_2, flags=2)

    return img_3, valid_matches_count/total_matches_count, keypoints_1, descriptors_1


def get_rectangle(vaid_matches, keypoints_2, width, height):
    # tl_x = width
    # tl_y = height
    # br_x = 0
    # br_y = 0
    # for match in vaid_matches:
    #     index = match.trainIdx
    #     keypoint = keypoints_2[index]
    #     if tl_x > keypoint.pt[0] or tl_y > keypoint.pt[1]:
    #         tl_x = keypoint.pt[0]
    #         tl_y = keypoint.pt[1]
    #     if br_x < keypoint.pt[0] or br_y < keypoint.pt[1]:
    #         br_x = keypoint.pt[0]
    #         br_y = keypoint.pt[1]
    
    tl_x = width
    tl_y = height
    br_x = -1
    br_y = -1
    for match in vaid_matches:
        index = match.trainIdx
        keypoint = keypoints_2[index]
        if tl_x > keypoint.pt[0]:
            tl_x = keypoint.pt[0]
        if br_x < keypoint.pt[0]:
            br_x = keypoint.pt[0]
        if tl_y > keypoint.pt[1]:
            tl_y = keypoint.pt[1]
        if br_y < keypoint.pt[1]:
            br_y = keypoint.pt[1]

    return (tl_x, tl_y), (br_x, br_y)

