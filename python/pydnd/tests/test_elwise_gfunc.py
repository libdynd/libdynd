import sys
import unittest
import pydnd as nd

class TestElwiseGFunc(unittest.TestCase):

    def test_square(self):
        # Tests some unary function evaluations
        x = nd.square(2)
        self.assertEqual(x.vals().dtype, nd.int32)
        self.assertEqual(x.as_py(), 4)

        x = nd.square(nd.arange(4))
        self.assertEqual(x.vals().dtype, nd.int32)
        self.assertEqual(x.as_py(), [0,1,4,9])

        x = nd.square(nd.arange(4).as_dtype(nd.float32)).as_dtype(nd.float64)
        self.assertEqual(x.vals().dtype, nd.float64)
        self.assertEqual(x.as_py(), [0.,1.,4.,9.])

        x = nd.square(nd.arange(4).as_dtype(nd.float32)).as_dtype(nd.float64)
        x = nd.square(x).as_dtype(nd.int64)
        self.assertEqual(x.vals().dtype, nd.int64)
        self.assertEqual(x.as_py(), [0,1,16,81])

    def test_add(self):
        # Tests some binary function evaluations
        x = nd.add(2, 3)
        self.assertEqual(x.vals().dtype, nd.int32)
        self.assertEqual(x.as_py(), 5)

        x = nd.add(nd.arange(4), 3)
        self.assertEqual(x.vals().dtype, nd.int32)
        self.assertEqual(x.as_py(), [3,4,5,6])

        x = nd.add(2, nd.arange(4))
        self.assertEqual(x.vals().dtype, nd.int32)
        self.assertEqual(x.as_py(), [2,3,4,5])

        x = nd.add(nd.arange(4), nd.arange(4))
        self.assertEqual(x.vals().dtype, nd.int32)
        self.assertEqual(x.as_py(), [0,2,4,6])

        x = nd.add(nd.arange(4).as_dtype(nd.float64), 3.5)
        self.assertEqual(x.vals().dtype, nd.float64)
        self.assertEqual(x.as_py(), [3.5,4.5,5.5,6.5])

        x = nd.add(nd.arange(4).as_dtype(nd.float64),
                nd.arange(4).as_dtype(nd.float64)).as_dtype(nd.int16)
        self.assertEqual(x.vals().dtype, nd.int16)
        self.assertEqual(x.as_py(), [0,2,4,6])

if __name__ == '__main__':
    unittest.main()

