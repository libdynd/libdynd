//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <vector>

struct parent {
  virtual ~parent();

  virtual int operator()() const = 0;
};

struct child : parent {
  int operator()() const;
};

extern std::vector<parent *> items;
