# String

Supporting strings well is tricky, as evidenced by the large numbers of
encodings and string API choices created over the years. Nowadays there
is no excuse not to go fully unicode, but that does not make the world
simple. The C++ standard does not presently have comprehensive unicode
support, though it finally has support for more types of unicode literals.
This leaves us in a position to adopt one of the existing unicode libraries
or implement much of the functionality ourselves.

* [Unicode in C++ CppCon 2014](https://www.youtube.com/watch?v=n0GK-9f4dl8)

## Design Choices

### One String Type

DyND contains a large number of integer, real, and complex number types,
something which is needed for being able to control memory usage and
performance while writing numerical codes. At first glance, it seems
natural to do a similar thing for strings, where the encoding and buffer
size are part of the type. This unfortunately can lead to a lot of
complexity. For example, supporting C-style null-terminated strings in
a fixed-size buffer is a natural desire. This is almost what NumPy does,
but NumPy allows the full buffer to be used with no NULL-terminator, so
NumPy's strings are a different variation.

The main reason to consider the approach of many types is performance,
to allow codes to be specialized for the various possible string types
being observed, and avoid extra copying. Given that strings are already
quite complex without this, that benefit can likely only be achieved at
the cost of having more bugs and putting less optimization effort into
the default string representation.

Other string storage choices can be viewed as the one string type
via expression types, which adapt between the representation.

### UTF-8 Encoding

There is a summary of a lot of discussion of UTF-8 versus other encodings
in the website manifesto http://utf8everywhere.org/. For example, if you
think Python's choice of having len(x) return the number of code points
is the "right answer", see the FAQ http://utf8everywhere.org/#myth.utf32.o1.
With the goal of being able to pump vast quantities of string data through
DyND, UTF-8 looks compelling.

### Code Unit API (Not Code Point)

Coming from a Python-centric background, it would be natural to adopt
Python's string APIs. Python's string API is based on code points, however,
and this conflicts with a choice of UTF-8 based strings. Go and Julia are
two more recently created languages, and they default to UTF-8 strings with
a code unit based API. I think this latter choice is more appropriate for DyND.

Consider the string "안녕!", which consists of the bytes
"ec 95 88 eb 85 95 21". It has three code points and seven bytes. In
Python, doing a find for "!" will return the value 2, indicating it is
the third code point. If we want to determine where in the UTF-8 string
that is, though, we need to walk through the characters from the beginning,
instead of instantly jumping to the correct offset. In the proposed code
unit API, find will return the value 6, which is the byte offset of the
character.

Both Julia (http://julia.readthedocs.org/en/latest/manual/strings/) and
Go (https://blog.golang.org/strings) have good explanations of how this works.

### Compatible with C-style String Algorithms

UTF-8 is designed to be compatible with most existing C-style string code,
and the only additional thing DyND has to do is guarantee a NULL terminator.
C++ does this in its std::string class as well.

### Value Semantics

Many array computational patterns as seen in systems like NumPy involve
initializing an array of values, then repeatedly modifying those values.
This generally works pretty reliably and predictably, because the numeric
types all have value semantics. If the string type has different semantics,
these patterns will fail or produce unexpected results.

### Exponential Growth on Append

One pattern to support well is repeated appending to a string. By keeping
a capacity member separate from the string size, an exponential capacity
growth can be used, similar to Python's list, or C++'s std::string and
std::vector. An example of what might be possible and efficient with
this (whether we want to do precisely this, we'll have to decide):

```python
>>> x = nd.array([['testing', 'one', 'two'], ['hi', 'there']],
                 type='2 * var * string')
>>> nd.join(' ', x, axis=1)
nd.array(['testing one two', 'hi there'],
         type = '2 * string')
```

### Fully Specified Memory Layout and Semantics

In many contexts, the details of something like a string type are kept
as an implementation detail, not something whose internals can be relied
on by other interoperating code. One of DyND's goals is to support
JIT compilations via systems like [Numba](http://numba.pydata.org/).
Such a JIT compiler should be able to create code that works with
DyND strings optimally, performing all the tricks that DyND as a C++
library can do, and perhaps even more.

### Small String Optimization

Memory allocations are something to be avoided in performance code, both
because of the time it takes to allocate and free as well as the cache
miss penalty incurred by pointer chasing. The small string optimization
provides a way to mitigate that cost when dealing with many small strings.

The small string optimization works by using something like a spare bit
in the representation to signal whether the data is stored in heap-allocated
memory, or right in the structure.

## Implementation details

The design choices made still give quite a bit of latitude in the actual
layout of data. Let's consider a few of the possibilities.

The smallest string storage would be a single pointer. The most natural
bit to use for the small string optimization is the lowest order bit,
because we can ensure the memory allocator always allocates aligned data.
In little-endian systems, this is the first byte, and it would be nicer
to use the last byte. To do this, the pointer could be shifted right
by one bit to free the uppermost bit for use as this signal. In this
approach, the allocated memory would begin with a size, capacity, then
the string data. A zero bit could be used to signal that the string
is stored inline, allowing that last byte to be zero and hence provide for
3 or 7 bytes of maximum string size on 32-bit and 64-bit systems,
respectively.

If we store the pointer followed by size in the representation, we get
more space for the inline string. It would be nice for the string size
to be trivially accessible in both the inline string and heap string cases,
so lets put the inline string size in the last byte. On a 64-bit system,
after also reserving the NULL terminator, this gives a maximum inline string
size of 14 bytes. On a 32-bit system, we have a choice, either to match 64-bit,
getting the same inline string size, or to go half the size, and only get a
6 byte maximum inline string. Matching the size seems reasonable, to get
similar performance on small strings as in 64 bits.

The result of this design is that the memory storage of a DyND string on
a little-endian architecture is

```cpp
struct string_memory_layout {
    int64_t m_pointer;
    int64_t m_size;

    bool is_inline() const {
        // If the highest bit is set
        return m_size < 0;
    }

    intptr_t inline_size() const {
        return (static_cast<uint64_t>(m_size) >> 56) & 0x7f;
    }

    intptr_t size() const {
        return is_inline() ? inline_size() : m_size;
    }

    intptr_t capacity() const {
        return is_inline() ? 15
                           : *reinterpret_cast<const intptr_t *>(m_pointer);
    }

    const char *data() const {
        return is_inline() ? reinterpret_cast<const char *>(this)
                           : reinterpret_cast<const char *>(m_pointer);
    }
};
```

And the heap allocated string data looks like

```
// For holding a string encoded into N bytes or fewer
template<int N>
struct string_heap_memory {
    intptr_t m_capacity; // Contains the value N+1
    char m_string[N+1];
};
```

The details for the string on a big-endian system will be a bit different,
and requires more work. DyND's testing infrastructure currently lacks any
big-endian systems, something we will need before we can reliably implement
it.
