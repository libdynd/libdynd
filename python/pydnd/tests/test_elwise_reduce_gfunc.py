import sys
import unittest
import pydnd as nd

class TestElwiseReduceGFunc(unittest.TestCase):

    def test_sum(self):
        # Tests an elementwise reduce gfunc with identity
        x = nd.sum([1,2,3,4,5]).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 15)

        x = nd.sum([1,2,3,4,5], associate='left').vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 15)

        x = nd.sum([1,2,3,4,5], associate='right').vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 15)

        x = nd.sum([1,2,3,4,5], keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [15])

        x = nd.sum([[1,2,3],[4,5,6]]).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 21)

        x = nd.sum([[1,2,3],[4,5,6]], axis=0).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [5,7,9])

        x = nd.sum([[1,2,3],[4,5,6]], axis=1).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [6,15])

        x = nd.sum([[1,2,3],[4,5,6]], axis=0, keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [[5,7,9]])

        x = nd.sum([[1,2,3],[4,5,6]], axis=1, keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [[6],[15]])

    def test_max(self):
        # Tests an elementwise reduce gfunc without an identity
        x = nd.max([1,2,8,4,5]).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 8)

        x = nd.max([1,2,8,4,5], associate='left').vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 8)

        x = nd.max([1,2,8,4,5], associate='right').vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 8)

        x = nd.max([1,2,8,4,5], keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [8])

        x = nd.max([[1,8,3],[4,5,6]]).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), 8)

        x = nd.max([[1,8,3],[4,5,6]], axis=0).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [4,8,6])

        x = nd.max([[1,8,3],[4,5,6]], axis=1).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [8,6])

        x = nd.max([[1,8,3],[4,5,6]], axis=0, keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [[4,8,6]])

        x = nd.max([[1,8,3],[4,5,6]], axis=1, keepdims=True).vals()
        self.assertEqual(x.dtype, nd.int32)
        self.assertEqual(x.as_py(), [[8],[6]])

if __name__ == '__main__':
    unittest.main()


