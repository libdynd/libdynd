import sys
import unittest
import pydnd as nd

class TestPythonScalar(unittest.TestCase):
    def test_bool(self):
        # Boolean true/false
        a = nd.ndarray(True)
        self.assertEqual(a.dtype, nd.bool)
        self.assertEqual(type(a.as_py()), bool)
        self.assertEqual(a.as_py(), True)
        a = nd.ndarray(False)
        self.assertEqual(a.dtype, nd.bool)
        self.assertEqual(type(a.as_py()), bool)
        self.assertEqual(a.as_py(), False)

    def test_int(self):
        # Integer that fits in 32 bits
        a = nd.ndarray(10)
        self.assertEqual(a.dtype, nd.int32)
        self.assertEqual(type(a.as_py()), int)
        self.assertEqual(a.as_py(), 10)
        a = nd.ndarray(-2000000000)
        self.assertEqual(a.dtype, nd.int32)
        self.assertEqual(type(a.as_py()), int)
        self.assertEqual(a.as_py(), -2000000000)

        # Integer that requires 64 bits
        a = nd.ndarray(2200000000)
        self.assertEqual(a.dtype, nd.int64)
        self.assertEqual(a.as_py(), 2200000000)
        a = nd.ndarray(-2200000000)
        self.assertEqual(a.dtype, nd.int64)
        self.assertEqual(a.as_py(), -2200000000)

    def test_float(self):
        # Floating point
        a = nd.ndarray(5.125)
        self.assertEqual(a.dtype, nd.float64)
        self.assertEqual(type(a.as_py()), float)
        self.assertEqual(a.as_py(), 5.125)

    def test_complex(self):
        # Complex floating point
        a = nd.ndarray(5.125 - 2.5j)
        self.assertEqual(a.dtype, nd.cfloat64)
        self.assertEqual(type(a.as_py()), complex)
        self.assertEqual(a.as_py(), 5.125 - 2.5j)

    def test_string(self):
        # String/Unicode TODO: Python 3 bytes becomes a bytes<> dtype
        a = nd.ndarray('abcdef')
        self.assertEqual(a.dtype, nd.make_fixedstring_dtype('ascii', 6))
        self.assertEqual(type(a.as_py()), unicode)
        self.assertEqual(a.as_py(), u'abcdef')
        a = nd.ndarray(u'abcdef')
        # Could be UTF 16 or 32 depending on the Python build configuration
        self.assertTrue(a.dtype == nd.make_fixedstring_dtype('utf_16', 6) or
                    a.dtype == nd.make_fixedstring_dtype('utf_32', 6))
        self.assertEqual(type(a.as_py()), unicode)
        self.assertEqual(a.as_py(), u'abcdef')

    def test_utf_encodings(self):
        # Ensure all of the UTF encodings work ok for a basic string
        x = u'\uc548\ub155 hello'
        # UTF-8
        a = nd.ndarray(x)
        a = a.as_dtype(nd.make_fixedstring_dtype('utf_8', 16))
        a = a.vals()
        self.assertEqual(a.dtype, nd.make_fixedstring_dtype('utf_8', 16))
        self.assertEqual(type(a.as_py()), unicode)
        self.assertEqual(a.as_py(), x)
        # UTF-16
        a = nd.ndarray(x)
        a = a.as_dtype(nd.make_fixedstring_dtype('utf_16', 8))
        a = a.vals()
        self.assertEqual(a.dtype, nd.make_fixedstring_dtype('utf_16', 8))
        self.assertEqual(type(a.as_py()), unicode)
        self.assertEqual(a.as_py(), x)
        # UTF-32
        a = nd.ndarray(x)
        a = a.as_dtype(nd.make_fixedstring_dtype('utf_32', 8))
        a = a.vals()
        self.assertEqual(a.dtype, nd.make_fixedstring_dtype('utf_32', 8))
        self.assertEqual(type(a.as_py()), unicode)
        self.assertEqual(a.as_py(), x)

    def test_len(self):
        # Can't get the length of a zero-dimensional ndarray
        a = nd.ndarray(10)
        self.assertRaises(TypeError, len, a)
