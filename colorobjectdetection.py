import numpy as np
import cv2 as cv
import glob


# red
red_lower_range_p = np.array([174, 102, 30])
red_upper_range_p = np.array([7, 255, 255])
# orange
orange_lower_range_p = np.array([10, 102, 30])
orange_upper_range_p = np.array([20, 255, 255])
# yellow
yellow_lower_range_p = np.array([19, 102, 30])
yellow_upper_range_p = np.array([31, 255, 255])
# green
green_lower_range_p = np.array([53, 102, 30])
green_upper_range_p = np.array([73, 255, 255])
# # green but bad
# green_lower_range_p = np.array([53, 102, 30])
# green_upper_range_p = np.array([82, 255, 255])
# blue
blue_lower_range_p = np.array([83, 102, 30])
blue_upper_range_p = np.array([103, 255, 255])
# purple
purple_lower_range_p = np.array([119, 102, 30])
purple_upper_range_p = np.array([131, 255, 255])
# black
black_lower_range_p = np.array([0, 0, 0])
black_upper_range_p = np.array([180, 64, 64])
# black_upper_range_p = np.array([180, 200, 64])
color_dict_p = {'Red': [red_lower_range_p, red_upper_range_p],
                # 'Orange': [orange_lower_range_p, orange_upper_range_p],
                'Yellow': [yellow_lower_range_p, yellow_upper_range_p],
                'Green': [green_lower_range_p, green_upper_range_p],
                'Blue': [blue_lower_range_p, blue_upper_range_p],
                'Purple': [purple_lower_range_p, purple_upper_range_p],
                'Black': [black_lower_range_p, black_upper_range_p]}

# red
red_lower_range_s = np.array([174, 50, 50])
red_upper_range_s = np.array([7, 255, 255])
# orange
orange_lower_range_s = np.array([10, 50, 50])
orange_upper_range_s = np.array([20, 255, 255])
# yellow
yellow_lower_range_s = np.array([9, 50, 50])
yellow_upper_range_s = np.array([24, 255, 255])
# green
green_lower_range_s = np.array([52, 50, 50])
green_upper_range_s = np.array([70, 255, 255])
# blue
blue_lower_range_s = np.array([94, 50, 50])
blue_upper_range_s = np.array([105, 255, 255])
# purple
purple_lower_range_s = np.array([107, 50, 50])
purple_upper_range_s = np.array([121, 255, 255])
# black
black_lower_range_s = np.array([0, 0, 0])
black_upper_range_s = np.array([180, 64, 64])
color_dict_s = {'Red': [red_lower_range_s, red_upper_range_s],
                # 'Orange': [orange_lower_range_s, orange_upper_range_s],
                'Yellow': [yellow_lower_range_s, yellow_upper_range_s],
                'Green': [green_lower_range_s, green_upper_range_s],
                'Blue': [blue_lower_range_s, blue_upper_range_s],
                'Purple': [purple_lower_range_s, purple_upper_range_s],
                'Black': [black_lower_range_s, black_upper_range_s]}


class ObjectColorDetector:
    def __init__(self, targets=None, PorS='p', K=7, T_sl=1000, T_su=4000, step_by_step_recording=False,
                 example_color='Yellow'):
        self.targets_c_h_s = {}
        self.kernelLength = K
        self.sizeThresholdDefault = [T_sl, T_su]
        self.openKernel = np.ones((self.kernelLength, self.kernelLength), np.uint8)
        self.closeKernel = np.ones((self.kernelLength * 2 + 1, self.kernelLength * 2 + 1), np.uint8)
        self.histograms_available = False
        if PorS == 'p':
            color_dict = color_dict_p
        elif PorS == 's':
            color_dict = color_dict_s
        else:
            print('assuming Pupil lab glasses')
            color_dict = color_dict_p
        sum_of_widths = 0
        if targets is None:
            keys = ['Red', 'Orange', 'yellow', 'Green', 'Blue', 'Purple', 'Black']
            target_dict = {}
            for key in keys:
                target_dict.update({key: [color_dict[key], [], self.sizeThresholdDefault]})
            print("No histograms available for precise search.")

            self.targets_c_h_s = target_dict
        else:
            for imagePath in targets:
                key = imagePath[:-6]
                target_img = cv.imread(imagePath)
                sum_of_widths += target_img.shape[1]
                target_hsv = cv.cvtColor(target_img, cv.COLOR_BGR2HSV)
                # histogram generation
                if key == 'Black':
                    histogram = cv.calcHist([target_hsv], [1, 2], None, [64, 64], [0, 256, 0, 256])
                else:
                    histogram = cv.calcHist([target_hsv], [0, 1], None, [45, 64], [0, 180, 0, 256])
                cv.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
                # size measurement
                if key == 'Red':
                    mask1 = cv.inRange(target_hsv, np.array([
                        0, color_dict[key][0][1], color_dict[key][0][2]]),
                                       color_dict[key][1])
                    mask2 = cv.inRange(target_hsv, color_dict[key][0],
                                       np.array(
                                           [180, color_dict[key][1][1], color_dict[key][1][2]]))
                    threshold_image = cv.bitwise_or(mask1, mask2)
                else:
                    threshold_image = cv.inRange(target_hsv, color_dict[key][0],
                                                 color_dict[key][1])
                opened_image = cv.morphologyEx(threshold_image, cv.MORPH_OPEN, self.openKernel)
                closed_image = cv.morphologyEx(opened_image, cv.MORPH_CLOSE, self.closeKernel)
                # cv.imshow(f"threshold, morpho image {key}", closed_image)
                # cv.waitKey()
                cn, h = cv.findContours(closed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                M00 = max(cv.moments(element)['m00'] for element in cn)
                sizeThresholds = [M00 / 2, M00 * 2]
                # update with final values
                self.targets_c_h_s.update({key: [color_dict[key], histogram, sizeThresholds]})
            self.histograms_available = True
            self.average_widths = np.floor(sum_of_widths / len(targets))
            self.step_by_step_recording = step_by_step_recording
            self.example_color = example_color

    def searchForAllObjects(self, image):
        best_matches_center_dict = {}
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        if self.step_by_step_recording:
            cv.imwrite("hsv_space_image.jpg", hsv_image)
        m_blurred_image = cv.medianBlur(hsv_image, self.kernelLength)
        if self.step_by_step_recording:
            cv.imwrite("median_blurred_image.jpg", m_blurred_image)
        for target in self.targets_c_h_s:
            best_matches_center_dict.update({target: self.searchForOneColor(m_blurred_image, target)})
        return best_matches_center_dict

    def searchForOneColor(self, hsv_blurred_image, key):
        if key == 'Red':
            mask1 = cv.inRange(hsv_blurred_image, np.array([
                0, self.targets_c_h_s[key][0][0][1], self.targets_c_h_s[key][0][0][2]]), self.targets_c_h_s[key][0][1])
            mask2 = cv.inRange(hsv_blurred_image, self.targets_c_h_s[key][0][0],
                               np.array([180, self.targets_c_h_s[key][0][1][1], self.targets_c_h_s[key][0][1][2]]))
            threshold_image = cv.bitwise_or(mask1, mask2)
        else:
            threshold_image = cv.inRange(hsv_blurred_image, self.targets_c_h_s[key][0][0],
                                         self.targets_c_h_s[key][0][1])
            if self.step_by_step_recording and key == self.example_color:
                cv.imwrite("hsv_threshold_image.jpg", threshold_image)
        opened_image = cv.morphologyEx(threshold_image, cv.MORPH_OPEN, self.openKernel)
        closed_image = cv.morphologyEx(opened_image, cv.MORPH_CLOSE, self.closeKernel)
        if self.step_by_step_recording and key == self.example_color:
            cv.imwrite("morphology_image.jpg", closed_image)
        cn, hierarchy = cv.findContours(closed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if self.step_by_step_recording and key == self.example_color:
            color_image_todrawon = cv.cvtColor(hsv_blurred_image, cv.COLOR_HSV2BGR)
            cv.drawContours(color_image_todrawon, cn, -1, (0, 255, 0), 3)
            cv.imwrite("contours_image.jpg", color_image_todrawon)
        possible_match_center = list()
        color_image_todrawon = cv.cvtColor(hsv_blurred_image, cv.COLOR_HSV2BGR)
        image_with_fewer_contours = np.copy(color_image_todrawon)
        for contour in cn:
            M = cv.moments(contour)
            if self.targets_c_h_s[key][2][0] < M['m00'] < self.targets_c_h_s[key][2][1]:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                possible_match_center.append([cx, cy])
                if self.step_by_step_recording and key == self.example_color:
                    cv.drawContours(image_with_fewer_contours, contour, -1, (0, 255, 0), 3)
                    cv.imwrite("fewer_contours_image.jpg", image_with_fewer_contours)
        if len(possible_match_center) > 1:
            if self.step_by_step_recording and key == self.example_color:
                color_image_todrawon = cv.cvtColor(hsv_blurred_image, cv.COLOR_HSV2BGR)
                for center in possible_match_center:
                    cv.circle(color_image_todrawon, center, 0, [255, 255, 255], 10)
                    cv.circle(color_image_todrawon, center, 2, 0, 3)
                cv.imwrite("centers_image.jpg", color_image_todrawon)
            similarities = {(-1, -1): "delete"}
            for detectionCoordinate in possible_match_center:
                lower_y = int((detectionCoordinate[0] - int(np.floor(self.average_widths / 2))))
                upper_y = int((detectionCoordinate[0] + int(np.floor((self.average_widths + 1) / 2))))
                lower_x = int((detectionCoordinate[1] - int(np.floor(self.average_widths / 2))))
                upper_x = int((detectionCoordinate[1] + int(np.floor((self.average_widths + 1) / 2))))
                ROI = hsv_blurred_image[lower_x:upper_x, lower_y:upper_y, :]
                detectionHistogram = cv.calcHist([ROI], [0, 1], None, [45, 64], [0, 180, 0, 256])
                if key == 'Black':
                    detectionHistogram = cv.calcHist([ROI], [1, 2], None, [64, 64], [0, 256, 0, 256])
                cv.normalize(detectionHistogram, detectionHistogram, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
                similarity_score = cv.compareHist(self.targets_c_h_s[key][1], detectionHistogram,
                                                  cv.HISTCMP_BHATTACHARYYA)
                # print(similarity_score)
                key_value_pair = {tuple(detectionCoordinate): similarity_score}
                similarities.update(key_value_pair)
            del similarities[(-1, -1)]
            best_match_center = min(similarities, key=similarities.get)
            best_match_center = [best_match_center[0], best_match_center[1]]  # this line is purely for type
            if self.step_by_step_recording and key == self.example_color:
                color_image_todrawon = cv.cvtColor(hsv_blurred_image, cv.COLOR_HSV2BGR)
                cv.circle(color_image_todrawon, best_match_center, 0, [255, 255, 255], 10)
                cv.circle(color_image_todrawon, best_match_center, 2, 0, 3)
                cv.imwrite("center_singular_image.jpg", color_image_todrawon)
            return best_match_center
        elif len(possible_match_center) == 1:
            return possible_match_center[0]
        else:
            return -1


