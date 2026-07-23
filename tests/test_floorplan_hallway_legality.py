from __future__ import annotations

import unittest

import numpy as np

from data_processing.floorplan_hallway_legality import (
    coordinates_to_pixels,
    hallway_flags,
    segment_nonhallway_ratio,
)


class FloorplanHallwayLegalityTest(unittest.TestCase):
    def test_coordinate_mapping_inverts_y(self) -> None:
        coordinates = np.array([[0.0, 0.0], [10.0, 20.0]])

        pixels_x, pixels_y, in_bounds = coordinates_to_pixels(
            coordinates,
            map_width_m=10.0,
            map_height_m=20.0,
            image_width=11,
            image_height=21,
            invert_y=True,
        )

        np.testing.assert_array_equal(pixels_x, np.array([0, 10]))
        np.testing.assert_array_equal(pixels_y, np.array([20, 0]))
        self.assertTrue(in_bounds.all())

    def test_nonhallway_pixel_is_not_legal(self) -> None:
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        image[1, 1] = np.array([195, 235, 245], dtype=np.uint8)
        coordinates = np.array([[1.0, 1.0], [0.0, 0.0]])

        flags, in_bounds = hallway_flags(
            image,
            coordinates,
            map_width_m=2.0,
            map_height_m=2.0,
            invert_y=True,
            max_rgb_channel_exclusive=80,
        )

        self.assertTrue(in_bounds.all())
        np.testing.assert_array_equal(flags, np.array([False, True]))

    def test_segment_reports_wall_crossing(self) -> None:
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        image[:, 2] = 200

        ratio = segment_nonhallway_ratio(
            image,
            start_xy=np.array([0.0, 1.0]),
            end_xy=np.array([4.0, 1.0]),
            map_width_m=4.0,
            map_height_m=2.0,
            invert_y=True,
            max_rgb_channel_exclusive=80,
        )

        self.assertAlmostEqual(ratio, 0.2)

    def test_out_of_bounds_point_is_not_legal(self) -> None:
        image = np.zeros((3, 3, 3), dtype=np.uint8)

        flags, in_bounds = hallway_flags(
            image,
            np.array([[3.0, 1.0]]),
            map_width_m=2.0,
            map_height_m=2.0,
            invert_y=True,
            max_rgb_channel_exclusive=80,
        )

        self.assertFalse(in_bounds[0])
        self.assertFalse(flags[0])


if __name__ == "__main__":
    unittest.main()
