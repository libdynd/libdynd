import sys
import unittest
import pydnd as nd
import numpy as np
from numpy.testing import *

class TestNumpyDTypeInterop(unittest.TestCase):
    def setUp(self):
        if sys.byteorder == 'little':
            self.nonnative = '>'
        else:
            self.nonnative = '<'

    def test_dtype_from_numpy_scalar_types(self):
        """Tests converting numpy scalar types to pydnd dtypes"""
        self.assertEqual(nd.bool, nd.dtype(np.bool))
        self.assertEqual(nd.int8, nd.dtype(np.int8))
        self.assertEqual(nd.int16, nd.dtype(np.int16))
        self.assertEqual(nd.int32, nd.dtype(np.int32))
        self.assertEqual(nd.int64, nd.dtype(np.int64))
        self.assertEqual(nd.uint8, nd.dtype(np.uint8))
        self.assertEqual(nd.uint16, nd.dtype(np.uint16))
        self.assertEqual(nd.uint32, nd.dtype(np.uint32))
        self.assertEqual(nd.uint64, nd.dtype(np.uint64))
        self.assertEqual(nd.float32, nd.dtype(np.float32))
        self.assertEqual(nd.float64, nd.dtype(np.float64))
        self.assertEqual(nd.cfloat32, nd.dtype(np.complex64))
        self.assertEqual(nd.cfloat64, nd.dtype(np.complex128))

    def test_dtype_from_numpy_dtype(self):
        """Tests converting numpy dtypes to pydnd dtypes"""
        # native byte order
        self.assertEqual(nd.bool, nd.dtype(np.dtype(np.bool)))
        self.assertEqual(nd.int8, nd.dtype(np.dtype(np.int8)))
        self.assertEqual(nd.int16, nd.dtype(np.dtype(np.int16)))
        self.assertEqual(nd.int32, nd.dtype(np.dtype(np.int32)))
        self.assertEqual(nd.int64, nd.dtype(np.dtype(np.int64)))
        self.assertEqual(nd.uint8, nd.dtype(np.dtype(np.uint8)))
        self.assertEqual(nd.uint16, nd.dtype(np.dtype(np.uint16)))
        self.assertEqual(nd.uint32, nd.dtype(np.dtype(np.uint32)))
        self.assertEqual(nd.uint64, nd.dtype(np.dtype(np.uint64)))
        self.assertEqual(nd.float32, nd.dtype(np.dtype(np.float32)))
        self.assertEqual(nd.float64, nd.dtype(np.dtype(np.float64)))
        self.assertEqual(nd.cfloat32, nd.dtype(np.dtype(np.complex64)))
        self.assertEqual(nd.cfloat64, nd.dtype(np.dtype(np.complex128)))
        self.assertEqual(nd.make_fixedstring_dtype('ascii', 10),
                    nd.dtype(np.dtype('S10')))
        self.assertEqual(nd.make_fixedstring_dtype('utf_32', 10),
                    nd.dtype(np.dtype('U10')))

        # non-native byte order
        nonnative = self.nonnative

        self.assertEqual(nd.make_byteswap_dtype(nd.int16),
                nd.dtype(np.dtype(nonnative + 'i2')))
        self.assertEqual(nd.make_byteswap_dtype(nd.int32),
                nd.dtype(np.dtype(nonnative + 'i4')))
        self.assertEqual(nd.make_byteswap_dtype(nd.int64),
                nd.dtype(np.dtype(nonnative + 'i8')))
        self.assertEqual(nd.make_byteswap_dtype(nd.uint16),
                nd.dtype(np.dtype(nonnative + 'u2')))
        self.assertEqual(nd.make_byteswap_dtype(nd.uint32),
                nd.dtype(np.dtype(nonnative + 'u4')))
        self.assertEqual(nd.make_byteswap_dtype(nd.uint64),
                nd.dtype(np.dtype(nonnative + 'u8')))
        self.assertEqual(nd.make_byteswap_dtype(nd.float32),
                nd.dtype(np.dtype(nonnative + 'f4')))
        self.assertEqual(nd.make_byteswap_dtype(nd.float64),
                nd.dtype(np.dtype(nonnative + 'f8')))
        self.assertEqual(nd.make_byteswap_dtype(nd.cfloat32),
                nd.dtype(np.dtype(nonnative + 'c8')))
        self.assertEqual(nd.make_byteswap_dtype(nd.cfloat64),
                nd.dtype(np.dtype(nonnative + 'c16')))

    def test_dnd_view_of_numpy_array(self):
        """Tests viewing a numpy array as a dnd ndarray"""
        nonnative = self.nonnative

        a = np.arange(10, dtype=np.int32)
        n = nd.ndarray(a)
        self.assertEqual(n.dtype, nd.int32)
        self.assertEqual(n.ndim, a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

        a = np.arange(12, dtype=(nonnative + 'i4')).reshape(3,4)
        n = nd.ndarray(a)
        self.assertEqual(n.dtype, nd.make_byteswap_dtype(nd.int32))
        self.assertEqual(n.ndim, a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

        a = np.arange(49, dtype='i1')
        a = a[1:].view(dtype=np.int32).reshape(4,3)
        n = nd.ndarray(a)
        self.assertEqual(n.dtype, nd.make_unaligned_dtype(nd.int32))
        self.assertEqual(n.ndim, a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

        a = np.arange(49, dtype='i1')
        a = a[1:].view(dtype=(nonnative + 'i4')).reshape(2,2,3)
        n = nd.ndarray(a)
        self.assertEqual(n.dtype,
                nd.make_unaligned_dtype(nd.make_byteswap_dtype(nd.int32)))
        self.assertEqual(n.ndim, a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

    def test_numpy_view_of_dnd_array(self):
        """Tests viewing a dnd ndarray as a numpy array"""
        nonnative = self.nonnative

        n = nd.ndarray(np.arange(10, dtype=np.int32))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(np.int32))
        self.assertTrue(a.flags.aligned)
        self.assertEqual(a.ndim, n.ndim)
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

        n = nd.ndarray(np.arange(12, dtype=(nonnative + 'i4')).reshape(3,4))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(nonnative + 'i4'))
        self.assertTrue(a.flags.aligned)
        self.assertEqual(a.ndim, n.ndim)
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

        n = nd.ndarray(np.arange(49, dtype='i1')[1:].view(dtype=np.int32).reshape(4,3))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(np.int32))
        self.assertFalse(a.flags.aligned)
        self.assertEqual(a.ndim, n.ndim)
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

        n = nd.ndarray(np.arange(49, dtype='i1')[1:].view(
                    dtype=(nonnative + 'i4')).reshape(2,2,3))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(nonnative + 'i4'))
        self.assertFalse(a.flags.aligned)
        self.assertEqual(a.ndim, n.ndim)
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

    def test_numpy_dnd_fixedstring_interop(self):
        """Tests converting fixed-size string arrays to/from numpy"""
        # ASCII Numpy -> dnd
        a = np.array(['abc', 'testing', 'array'])
        b = nd.ndarray(a)
        self.assertEqual(nd.make_fixedstring_dtype('ascii', 7), b.dtype)
        self.assertEqual(b.dtype, nd.dtype(a.dtype))

        # ASCII dnd -> Numpy
        c = np.asarray(b)
        self.assertEqual(a.dtype, c.dtype)
        assert_array_equal(a, c)
        # verify 'a' and 'c' are looking at the same data
        a[1] = 'modify'
        assert_array_equal(a, c)

        # ASCII dnd -> UTF32 dnd
        b_u = b.as_dtype(nd.make_fixedstring_dtype('utf_32', 7))
        self.assertEqual(
                nd.make_convert_dtype(
                    nd.make_fixedstring_dtype('utf_32', 7),
                    nd.make_fixedstring_dtype('ascii', 7)),
                b_u.dtype)
        # Evaluate to its value array
        b_u = b_u.vals()
        self.assertEqual(
                nd.make_fixedstring_dtype('utf_32', 7),
                b_u.dtype)

        # UTF32 dnd -> Numpy
        c_u = np.asarray(b_u)
        self.assertEqual(b_u.dtype, nd.dtype(c_u.dtype))
        assert_array_equal(a, c_u)
        # 'a' and 'c_u' are not looking at the same data
        a[1] = 'diff'
        self.assertFalse(np.all(a == c_u))

