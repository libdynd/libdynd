import sys
import unittest
import pydnd as nd

class TestDType(unittest.TestCase):

    def test_bool_dtype_properties(self):
        self.assertEqual(type(nd.bool), nd.dtype)
        self.assertEqual(str(nd.bool), 'bool')
        self.assertEqual(nd.bool.element_size, 1)
        self.assertEqual(nd.bool.alignment, 1)

    def test_int_dtype_properties(self):
        self.assertEqual(type(nd.int8), nd.dtype)
        self.assertEqual(str(nd.int8), 'int8')
        self.assertEqual(nd.int8.element_size, 1)
        self.assertEqual(nd.int8.alignment, 1)

        self.assertEqual(type(nd.int16), nd.dtype)
        self.assertEqual(str(nd.int16), 'int16')
        self.assertEqual(nd.int16.element_size, 2)
        self.assertEqual(nd.int16.alignment, 2)

        self.assertEqual(type(nd.int32), nd.dtype)
        self.assertEqual(str(nd.int32), 'int32')
        self.assertEqual(nd.int32.element_size, 4)
        self.assertEqual(nd.int32.alignment, 4)

        self.assertEqual(type(nd.int64), nd.dtype)
        self.assertEqual(str(nd.int64), 'int64')
        self.assertEqual(nd.int64.element_size, 8)
        self.assertEqual(nd.int64.alignment, 8)

    def test_uint_dtype_properties(self):
        self.assertEqual(type(nd.uint8), nd.dtype)
        self.assertEqual(str(nd.uint8), 'uint8')
        self.assertEqual(nd.uint8.element_size, 1)
        self.assertEqual(nd.uint8.alignment, 1)

        self.assertEqual(type(nd.uint16), nd.dtype)
        self.assertEqual(str(nd.uint16), 'uint16')
        self.assertEqual(nd.uint16.element_size, 2)
        self.assertEqual(nd.uint16.alignment, 2)

        self.assertEqual(type(nd.uint32), nd.dtype)
        self.assertEqual(str(nd.uint32), 'uint32')
        self.assertEqual(nd.uint32.element_size, 4)
        self.assertEqual(nd.uint32.alignment, 4)

        self.assertEqual(type(nd.uint64), nd.dtype)
        self.assertEqual(str(nd.uint64), 'uint64')
        self.assertEqual(nd.uint64.element_size, 8)
        self.assertEqual(nd.uint64.alignment, 8)

    def test_float_dtype_properties(self):
        self.assertEqual(type(nd.float32), nd.dtype)
        self.assertEqual(str(nd.float32), 'float32')
        self.assertEqual(nd.float32.element_size, 4)
        self.assertEqual(nd.float32.alignment, 4)

        self.assertEqual(type(nd.float64), nd.dtype)
        self.assertEqual(str(nd.float64), 'float64')
        self.assertEqual(nd.float64.element_size, 8)
        self.assertEqual(nd.float64.alignment, 8)

    def test_complex_dtype_properties(self):
        self.assertEqual(type(nd.cfloat32), nd.dtype)
        self.assertEqual(str(nd.cfloat32), 'complex<float32>')
        self.assertEqual(nd.cfloat32.element_size, 8)
        self.assertEqual(nd.cfloat32.alignment, 4)

        self.assertEqual(type(nd.cfloat64), nd.dtype)
        self.assertEqual(str(nd.cfloat64), 'complex<float64>')
        self.assertEqual(nd.cfloat64.element_size, 16)
        self.assertEqual(nd.cfloat64.alignment, 8)

    def test_fixedstring_dtype_properties(self):
        d = nd.make_fixedstring_dtype('ascii', 10)
        self.assertEqual(str(d), 'fixedstring<ascii,10>')
        self.assertEqual(d.element_size, 10)
        self.assertEqual(d.alignment, 1)
        self.assertEqual(d.string_encoding, 'ascii')

        d = nd.make_fixedstring_dtype('ucs_2', 10)
        self.assertEqual(str(d), 'fixedstring<ucs_2,10>')
        self.assertEqual(d.element_size, 20)
        self.assertEqual(d.alignment, 2)
        self.assertEqual(d.string_encoding, 'ucs_2')

        d = nd.make_fixedstring_dtype('utf_8', 10)
        self.assertEqual(str(d), 'fixedstring<utf_8,10>')
        self.assertEqual(d.element_size, 10)
        self.assertEqual(d.alignment, 1)
        self.assertEqual(d.string_encoding, 'utf_8')

        d = nd.make_fixedstring_dtype('utf_16', 10)
        self.assertEqual(str(d), 'fixedstring<utf_16,10>')
        self.assertEqual(d.element_size, 20)
        self.assertEqual(d.alignment, 2)
        self.assertEqual(d.string_encoding, 'utf_16')

        d = nd.make_fixedstring_dtype('utf_32', 10)
        self.assertEqual(str(d), 'fixedstring<utf_32,10>')
        self.assertEqual(d.element_size, 40)
        self.assertEqual(d.alignment, 4)
        self.assertEqual(d.string_encoding, 'utf_32')

    def test_scalar_dtypes(self):
        self.assertEqual(nd.bool, nd.dtype(bool))
        self.assertEqual(nd.int32, nd.dtype(int))
        self.assertEqual(nd.float64, nd.dtype(float))
        self.assertEqual(nd.cfloat64, nd.dtype(complex))

    def test_fixedbytes_dtype(self):
        d = nd.make_fixedbytes_dtype(4, 4)
        self.assertEqual(str(d), 'fixedbytes<4,4>')
        self.assertEqual(d.element_size, 4)
        self.assertEqual(d.alignment, 4)

        d = nd.make_fixedbytes_dtype(9, 1)
        self.assertEqual(str(d), 'fixedbytes<9,1>')
        self.assertEqual(d.element_size, 9)
        self.assertEqual(d.alignment, 1)

        # Alignment must not be greater than element_size
        self.assertRaises(RuntimeError, nd.make_fixedbytes_dtype, 1, 2)
        # Alignment must be a power of 2
        self.assertRaises(RuntimeError, nd.make_fixedbytes_dtype, 6, 3)
        # Alignment must divide into the element_size
        self.assertRaises(RuntimeError, nd.make_fixedbytes_dtype, 6, 4)
