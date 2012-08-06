import sys
import unittest
import pydnd as nd

class TestPythonScalar(unittest.TestCase):
    def test_bool(self):
        lst = [True, False, True, True]
        a = nd.ndarray(lst)
        self.assertEqual(a.dtype, nd.bool)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(a.as_py(), lst)

        lst = [[True, True], [False, False], [True, False]]
        a = nd.ndarray(lst)
        self.assertEqual(a.dtype, nd.bool)
        self.assertEqual(a.shape, (3,2))
        self.assertEqual(a.as_py(), lst)

    def test_int32(self):
        lst = [0, 100, 2000000000, -1000000000]
        a = nd.ndarray(lst)
        self.assertEqual(a.dtype, nd.int32)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(a.as_py(), lst)

        lst = [[100, 103, -20], [-30, 0, 1000000]]
        a = nd.ndarray(lst)
        self.assertEqual(a.dtype, nd.int32)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.as_py(), lst)

    def test_int64(self):
        lst = [0, 100, 20000000000, -1000000000]
        a = nd.ndarray(lst)
        self.assertEqual(a.dtype, nd.int64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(a.as_py(), lst)

        lst = [[100, 103, -20], [-30, 0, 1000000000000]]
        a = nd.ndarray(lst)
        self.assertEqual(a.dtype, nd.int64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.as_py(), lst)

    def test_float64(self):
        lst = [0, 100.0, 1e25, -1000000000]
        a = nd.ndarray(lst)
        self.assertEqual(a.dtype, nd.float64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(a.as_py(), lst)

        lst = [[100, 103, -20], [-30, 0.0125, 1000000000000]]
        a = nd.ndarray(lst)
        self.assertEqual(a.dtype, nd.float64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.as_py(), lst)

    def test_float64(self):
        lst = [0, 100.0, 1e25, -1000000000+3j]
        a = nd.ndarray(lst)
        self.assertEqual(a.dtype, nd.complex_float64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(a.as_py(), lst)

        lst = [[100, 103j, -20], [-30, 0.0125, 1000000000000]]
        a = nd.ndarray(lst)
        self.assertEqual(a.dtype, nd.complex_float64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.as_py(), lst)


if __name__ == '__main__':
    unittest.main()
