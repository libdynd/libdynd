import sys
import unittest
import pydnd as nd
import numpy as np

class TestNumpyDTypeInterop(unittest.TestCase):

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

        # non-native byte order
        if sys.byteorder == 'little':
            eindicator = '>'
        else:
            eindicator = '<'

        self.assertEqual(nd.make_byteswap_dtype(nd.int16),
                nd.dtype(np.dtype(eindicator + 'i2')))
        self.assertEqual(nd.make_byteswap_dtype(nd.int32),
                nd.dtype(np.dtype(eindicator + 'i4')))
        self.assertEqual(nd.make_byteswap_dtype(nd.int64),
                nd.dtype(np.dtype(eindicator + 'i8')))
        self.assertEqual(nd.make_byteswap_dtype(nd.uint16),
                nd.dtype(np.dtype(eindicator + 'u2')))
        self.assertEqual(nd.make_byteswap_dtype(nd.uint32),
                nd.dtype(np.dtype(eindicator + 'u4')))
        self.assertEqual(nd.make_byteswap_dtype(nd.uint64),
                nd.dtype(np.dtype(eindicator + 'u8')))
        self.assertEqual(nd.make_byteswap_dtype(nd.float32),
                nd.dtype(np.dtype(eindicator + 'f4')))
        self.assertEqual(nd.make_byteswap_dtype(nd.float64),
                nd.dtype(np.dtype(eindicator + 'f8')))
        self.assertEqual(nd.make_byteswap_dtype(nd.cfloat32),
                nd.dtype(np.dtype(eindicator + 'c8')))
        self.assertEqual(nd.make_byteswap_dtype(nd.cfloat64),
                nd.dtype(np.dtype(eindicator + 'c16')))
