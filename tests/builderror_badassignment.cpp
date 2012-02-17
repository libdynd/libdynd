// This tests for an expected build error
#include <dnd/ndarray.hpp>

using namespace dnd;

int main()
{
    //ndarray a(10, make_dtype<int>());

    // This should fail, because only by-reference assignments
    // are allowed on raw ndarrays, but the programmer would
    // expect a by-value assignment here. This is made to fail
    // by having the indexing operator return a 'const ndarray',
    // which then cannot be assigned to.
    //a(1).vals() = 100;

    // The correct syntax for the above intention is:
    // a(1).vals() = 100;
}
