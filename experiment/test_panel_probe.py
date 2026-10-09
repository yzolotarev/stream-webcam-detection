"""Synthetic unit tests; no videos, cached results, or annotations are read."""
import copy
import unittest
from unittest.mock import Mock

import numpy as np

import panel_probe as probe


CORNERS = ('tl', 'tr', 'bl', 'br')


def rectangle(width=80, height=60, value=210):
    image = np.full((180, 320, 3), 20, np.uint8)
    image[:height, :width] = value
    return image


def seed(count=4, corner='tl'):
    box = probe.panel_box(0.04, 0.06, corner, 1, 1)
    box[0] += 0.03 if corner[1] == 'l' else -0.03
    box[1] += 0.04 if corner[0] == 't' else -0.04
    return {'count': count, 'median_face_bbox': box,
            'frame_indices': list(range(count))}


def video(*clusters):
    return {'clusters': {'yunet': list(clusters)}}


def sample(time=0):
    return {'index': 0, 'sample_id': 'synthetic', 'requested_s': time,
            'actual_s': time, 'yunet': []}


def proposal(score=0.5, side=0.5, distance=0.0):
    return {'w': 80, 'h': 60, 'score': score, 'vertical': side,
            'horizontal': side, 'appearance_distance': distance}


class GeometryTests(unittest.TestCase):
    def test_orientation_corners_and_involution(self):
        image = np.arange(5 * 7 * 3).reshape(5, 7, 3)
        for corner in CORNERS:
            with self.subTest(corner=corner):
                oriented = probe.orient(image, corner)
                y = 0 if corner[0] == 't' else -1
                x = 0 if corner[1] == 'l' else -1
                np.testing.assert_array_equal(oriented[0, 0], image[y, x])
                np.testing.assert_array_equal(probe.orient(oriented, corner), image)
                self.assertTrue(oriented.flags.c_contiguous)

    def test_invalid_orientation(self):
        for corner in ('lt', '', 'center'):
            with self.subTest(corner=corner), self.assertRaises(ValueError):
                probe.orient(rectangle(), corner)

    def test_face_corner_uses_center_and_midpoint_ties(self):
        for corner in CORNERS:
            self.assertEqual(probe.face_corner(seed(corner=corner)['median_face_bbox']), corner)
        self.assertEqual(probe.face_corner([0.4, 0.4, 0.2, 0.2]), 'br')
        self.assertEqual(probe.face_corner([0.49, 0.01, 0.1, 0.1]), 'tr')

    def test_canonical_face_all_corners_and_extra_fields(self):
        for corner in CORNERS:
            with self.subTest(corner=corner):
                box = seed(corner=corner)['median_face_bbox'] + [0.99]
                np.testing.assert_allclose(probe.canonical_face(box, corner, 320, 180),
                                           [9.6, 7.2, 12.8, 10.8])

    def test_panel_boxes_are_half_open_xywh(self):
        expected = ([0, 0, 80, 60], [240, 0, 80, 60],
                    [0, 120, 80, 60], [240, 120, 80, 60])
        for corner, box in zip(CORNERS, expected):
            self.assertEqual(probe.panel_box(80, 60, corner, 320, 180), box)

    def test_allowed_map_contains_face_and_rejects_outsize_rectangles(self):
        mask = probe.allowed_map((180, 320), [10, 10, 20, 20])
        self.assertTrue(mask[60, 80])
        self.assertFalse(mask[30, 80])
        self.assertFalse(mask[60, 30])
        self.assertFalse(mask[120, 80])
        self.assertFalse(mask[60, 200])
        self.assertFalse(probe.allowed_map((180, 320), [200, 140, 30, 30]).any())

    def test_face_in_candidate_is_center_test_not_containment(self):
        self.assertFalse(probe.face_inside([], [0, 0, 0.25, 0.3]))
        self.assertTrue(probe.face_inside([[0.2, 0.2, 0.1, 0.1]], [0, 0, 0.25, 0.3]))
        self.assertFalse(probe.face_inside([[0.3, 0.2, 0.1, 0.1]], [0, 0, 0.25, 0.3]))


class BorderAndSeedTests(unittest.TestCase):
    def test_uniform_image_has_no_border_proposals(self):
        maps = probe.border_maps(np.full((180, 320, 3), 100, np.uint8))
        for array in maps:
            self.assertEqual(array.dtype, np.float32)
            self.assertFalse(array.any())
        self.assertEqual(probe.peaks(maps[0], np.ones((180, 320), bool)), [])

    def test_synthetic_border_recovery_all_corners(self):
        mask = probe.allowed_map((180, 320), [10, 10, 20, 20])
        for corner in CORNERS:
            with self.subTest(corner=corner):
                image = probe.orient(rectangle(), corner)
                maps = probe.border_maps(probe.orient(image, corner))
                best = probe.peaks(maps[0], mask, count=1)[0]
                # The two-pixel finite difference can put an edge one pixel either side.
                self.assertLessEqual(abs(best['w'] - 80), 1)
                self.assertLessEqual(abs(best['h'] - 60), 1)
                self.assertGreater(best['score'], probe.CONFIG['seed_score_min'])

    def test_peaks_mask_nonpositive_and_suppression(self):
        scores = np.zeros((20, 20), np.float32)
        scores[0, 0], scores[1, 1], scores[15, 15] = 0.9, 0.8, 0.7
        original = scores.copy()
        mask = np.ones_like(scores, bool)
        found = probe.peaks(scores, mask, radius=2)
        self.assertEqual([(p['w'], p['h']) for p in found], [(0, 0), (15, 15)])
        np.testing.assert_array_equal(scores, original)
        self.assertEqual(probe.peaks(scores, ~mask), [])
        self.assertEqual(probe.peaks(scores, mask, count=0), [])

    def test_local_search_limits_and_missing_evidence(self):
        scores = np.zeros((180, 320), np.float32)
        scores[60, 80], scores[70, 95], scores[100, 140] = 0.4, 0.8, 1
        maps = (scores, scores.copy(), scores.copy())
        found = probe.local_candidates(maps, {'w': 80, 'h': 60}, np.ones_like(scores, bool))
        self.assertEqual((found['stable']['w'], found['stable']['h']), (80, 60))
        self.assertEqual((found['changed']['w'], found['changed']['h']), (95, 70))
        self.assertEqual(probe.local_candidates(maps, {'w': 80, 'h': 60}, np.zeros_like(scores, bool)), {})

    def test_empty_and_insufficient_seeds(self):
        for clusters in ((), (seed(2),)):
            with self.subTest(clusters=clusters):
                self.assertEqual(probe.build_model(video(*clusters), [rectangle()] * 4),
                                 (None, 'insufficient_recurring_faces'))
        self.assertEqual(probe.build_model(video(seed(3)), [rectangle()] * 40),
                         (None, 'insufficient_recurring_faces'))
        self.assertEqual(probe.build_model(video(), []), (None, 'insufficient_recurring_faces'))

    def test_competing_face_seeds_at_threshold(self):
        self.assertEqual(probe.build_model(video(seed(5), seed(4)), [rectangle()] * 5),
                         (None, 'competing_face_clusters'))

    def test_no_rectangle_seed(self):
        self.assertEqual(probe.build_model(video(seed()), [np.zeros((180, 320, 3), np.uint8)] * 4),
                         (None, 'no_supported_corner_rectangle'))

    def test_competing_panel_geometries(self):
        image = rectangle(110, 85, 120)
        image[:50, :65] = 230
        self.assertEqual(probe.build_model(video(seed()), [image] * 4),
                         (None, 'competing_panel_geometries'))

    def test_too_few_appearance_references(self):
        images = [rectangle()] * 2 + [np.full((180, 320, 3), 20, np.uint8)]
        self.assertEqual(probe.build_model(video(seed(3)), images),
                         (None, 'insufficient_appearance_references'))

    def test_synthetic_model_builds_without_mutating_seed(self):
        source = video(seed())
        before = copy.deepcopy(source)
        model, error = probe.build_model(source, [rectangle()] * 4)
        self.assertIsNone(error)
        self.assertEqual(model['size'], (320, 180))
        self.assertEqual(len(model['bank']), 4)
        self.assertEqual(source, before)


class ProposalTests(unittest.TestCase):
    def test_appearance_normalization_and_self_exclusion(self):
        feature = probe.appearance(np.full((20, 30, 3), 255, np.uint8), {'w': 30, 'h': 20})
        self.assertEqual(feature.shape, (16, 16, 3))
        self.assertEqual(feature.dtype, np.float32)
        np.testing.assert_array_equal(feature, np.ones_like(feature))
        bank = [(0, feature), (1, np.zeros_like(feature))]
        self.assertEqual(probe.appearance_distance(feature, bank, None), 0)
        self.assertEqual(probe.appearance_distance(feature, bank, 0), 1)
        self.assertEqual(probe.appearance_distance(feature, [], None), 1)
        self.assertEqual(probe.appearance_distance(feature, bank[:1], 0), 1)

    def test_stable_acceptance_at_thresholds(self):
        p = proposal(probe.CONFIG['stable_score_min'], probe.CONFIG['stable_side_min'],
                     probe.CONFIG['appearance_rmse_max'])
        selected, status, reason = probe.select_proposal({'stable': p})
        self.assertIs(selected, p)
        self.assertEqual((status, reason), ('detected', 'borders_and_appearance'))

    def test_changed_requires_strict_gain_and_falls_back_to_stable(self):
        stable = proposal(0.6)
        changed = proposal(0.6 * probe.CONFIG['change_gain_min'])
        self.assertIs(probe.select_proposal({'stable': stable, 'changed': changed})[0], stable)
        changed['score'] += 0.001
        result = probe.select_proposal({'stable': stable, 'changed': changed})
        self.assertIs(result[0], changed)
        self.assertEqual(result[2], 'relocalized_borders_and_appearance')

    def test_changed_can_be_accepted_without_stable(self):
        p = proposal(probe.CONFIG['changed_score_min'], probe.CONFIG['changed_side_min'])
        self.assertIs(probe.select_proposal({'changed': p})[0], p)

    def test_strong_borders_with_appearance_mismatch_abstain(self):
        self.assertEqual(probe.select_proposal({'stable': proposal(0.9, 0.9, 1),
                                                'changed': proposal(0.95, 0.95, 1)}),
                         (None, 'ambiguous', 'appearance_mismatch'))

    def test_bad_changed_appearance_does_not_displace_stable(self):
        stable = proposal()
        self.assertIs(probe.select_proposal({'stable': stable, 'changed': proposal(0.9, 0.9, 1)})[0], stable)

    def test_one_weak_side_abstains_even_with_high_score(self):
        p = proposal(0.9)
        p['horizontal'] = 0.01
        self.assertEqual(probe.select_proposal({'stable': p, 'changed': p}),
                         (None, 'ambiguous', 'weak_or_conflicting_borders'))

    def test_empty_and_weak_proposals_not_detected(self):
        for proposals in ({}, {'stable': proposal(0.01)}):
            self.assertEqual(probe.select_proposal(proposals),
                             (None, 'not_detected', 'insufficient_border_evidence'))


class InferenceTests(unittest.TestCase):
    def test_all_corners_original_pixel_scaling_and_normalization(self):
        for corner in CORNERS:
            with self.subTest(corner=corner):
                image = probe.orient(rectangle(), corner)
                model, error = probe.build_model(video(seed(corner=corner)), [image] * 4)
                self.assertIsNone(error)
                row = probe.infer(image, sample(), model, (1001, 563), exclude_index=0)
                self.assertEqual(row['status'], 'detected')
                selected = row['evidence']['stable']
                pw, ph = round(selected['w'] * 1001 / 320), round(selected['h'] * 563 / 180)
                self.assertEqual(row['bbox_px'], probe.panel_box(pw, ph, corner, 1001, 563))
                np.testing.assert_allclose(list(row['webcam'].values()),
                                           np.array(row['bbox_px']) / [1001, 563, 1001, 563])
                self.assertFalse(row['face_in_candidate'])
                self.assertTrue(all(0 <= v <= 1 for v in row['webcam'].values()))

    def test_disappearance_does_not_carry_detection_bbox(self):
        image = rectangle()
        model, _ = probe.build_model(video(seed()), [image] * 4)
        self.assertEqual(probe.infer(image, sample(), model, (320, 180))['status'], 'detected')
        missing = probe.infer(np.full_like(image, 20), sample(1), model, (320, 180))
        self.assertEqual(missing['status'], 'not_detected')
        self.assertIsNone(missing['bbox_px'])
        self.assertIsNone(missing['webcam'])
        self.assertIn('candidate_bbox_normalized', missing)
        self.assertEqual(probe.infer(image, sample(2), model, (320, 180))['status'], 'detected')

    def test_appearance_mismatch_in_real_image_path(self):
        model, _ = probe.build_model(video(seed()), [rectangle()] * 4)
        row = probe.infer(rectangle(value=70), sample(), model, (320, 180))
        self.assertEqual((row['status'], row['reason']), ('ambiguous', 'appearance_mismatch'))
        self.assertIsNone(row['bbox_px'])
        self.assertIsNone(row['webcam'])

    def test_cached_proposals_not_mutated_and_self_match_excluded(self):
        image = rectangle()
        model, _ = probe.build_model(video(seed()), [image] * 4)
        cached = model['locals'][0]
        before = copy.deepcopy(cached)
        model['bank'] = model['bank'][:1]
        row = probe.infer(image, sample(), model, (320, 180), proposals=cached, exclude_index=0)
        self.assertEqual(row['reason'], 'appearance_mismatch')
        self.assertEqual(cached, before)

    def test_missing_model_abstains(self):
        row = probe.infer(rectangle(), sample(), None, (320, 180), 'competing_face_clusters')
        self.assertEqual(row['status'], 'ambiguous')
        self.assertEqual(row['reason'], 'competing_face_clusters')
        self.assertIsNone(row['bbox_px'])
        self.assertIsNone(row['webcam'])


class SamplingTests(unittest.TestCase):
    @staticmethod
    def row(t, detected=False, face=False):
        return {'actual_s': t, 'status': 'detected' if detected else 'ambiguous',
                'face_in_candidate': face}

    def test_dense_empty_healthy_and_zero_budget(self):
        self.assertEqual(probe.dense_timestamps([], 100), [])
        self.assertEqual(probe.dense_timestamps([self.row(5, True, True)], 100), [])
        self.assertEqual(probe.dense_timestamps([self.row(5)], 100, maximum_centers=0), [])
        self.assertEqual(probe.dense_timestamps([self.row(0)], 0), [])

    def test_dense_budget_sorted_unique_range_and_existing_exclusion(self):
        frames = [self.row(t) for t in range(0, 101, 5)]
        times = probe.dense_timestamps(frames, 100)
        self.assertLessEqual(len(times), 6 * 4)
        self.assertEqual(times, sorted(set(times)))
        self.assertTrue(all(0 <= t < 100 for t in times))
        self.assertTrue(all(abs(t - f['actual_s']) > 0.1 for t in times for f in frames))
        self.assertIn(1, times)
        self.assertIn(99, times)

    def test_dense_detected_without_face_is_suspect_and_offsets_deduplicate(self):
        frames = [self.row(0, True), self.row(2, True)]
        self.assertEqual(probe.dense_timestamps(frames, 5), [1, 3, 4])

    def test_dense_avoids_nearby_existing_sample(self):
        frames = [self.row(5), self.row(6.05, True, True)]
        self.assertEqual(probe.dense_timestamps(frames, 10), [3, 4, 7])

    def test_dense_rounded_timestamp_stays_before_duration(self):
        # Bounds must apply to the emitted seek time, not just the unrounded sum.
        duration = 1.0008
        times = probe.dense_timestamps([self.row(0.0006)], duration)
        self.assertTrue(all(0 <= t < duration for t in times), times)

    def test_read_frame_resize_actual_time_and_seek_units(self):
        cap = Mock()
        cap.read.return_value = (True, np.zeros((240, 320, 3), np.uint8))
        cap.get.return_value = 2050
        image, actual = probe.read_frame(cap, 2)
        cap.set.assert_called_once_with(probe.cv2.CAP_PROP_POS_MSEC, 2000)
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertEqual(actual, 2.05)

    def test_read_frame_rejects_decode_failure_and_inaccurate_seek(self):
        cap = Mock()
        cap.read.return_value = (False, None)
        with self.assertRaisesRegex(RuntimeError, 'Decode failed'):
            probe.read_frame(cap, 2)
        cap.read.return_value = (True, rectangle())
        cap.get.return_value = 2300
        with self.assertRaisesRegex(RuntimeError, 'Inaccurate seek'):
            probe.read_frame(cap, 2)


if __name__ == '__main__':
    unittest.main()
