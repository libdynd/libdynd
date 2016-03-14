#include <dispatcher.hpp>

parent::~parent() {}

int child::operator()() const { return 0; }

std::vector<parent *> items = {new child()};
