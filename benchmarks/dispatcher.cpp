#include <dispatcher.hpp>

parent::~parent() {}

int child::operator()() const { return 0; }

parent *item = new child();
