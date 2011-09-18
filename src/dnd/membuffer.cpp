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

membuffer::membuffer(const dtype& d, intptr_t size)
    : m_dtype(d), m_size(size), m_data(new char[size * d.itemsize()])
{
}

membuffer::~membuffer()
{
    delete[] m_data;
}
