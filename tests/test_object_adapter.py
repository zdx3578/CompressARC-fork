import unittest
import numpy as np
from utils.object_adapter import dedup_masks, _flood_fill_holes

class TestObjectAdapter(unittest.TestCase):
    def test_dedup_masks(self):
        masks = [
            np.array([[1,0],[1,0]], dtype=bool),
            np.array([[1,0],[1,0]], dtype=bool),
            np.array([[0,1],[0,1]], dtype=bool),
        ]
        objs = [{'id':0},{'id':1},{'id':2}]
        new_objs, new_masks, keep = dedup_masks(objs, masks, iou_thr=0.9)
        self.assertEqual(len(new_masks), 2)
        self.assertEqual(len(new_objs), 2)
        self.assertListEqual(keep, [0,2])

    def test_flood_fill_holes(self):
        mask = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=bool)
        self.assertEqual(_flood_fill_holes(mask), 1)
        mask2 = np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,1,1,1]], dtype=bool)
        self.assertEqual(_flood_fill_holes(mask2), 1)

if __name__ == '__main__':
    unittest.main()
