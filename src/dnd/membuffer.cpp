//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#include <dnd/membuffer.hpp>

#include <stdexcept>

using namespace std;
using namespace dnd;

membuffer::membuffer(const dtype& d, intptr_t num_elements)
    : m_dtype(d), m_data(new char[num_elements * d.itemsize()]),
        m_size(num_elements)
{
}

membuffer::~membuffer()
{
    delete[] m_data;
}
