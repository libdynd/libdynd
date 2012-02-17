// This build test is to confirm the build environment for the
// compile error tests can successfully include ndarray.hpp
// and make an ndarray object

#include <dnd/ndarray.hpp>

using namespace std;
using namespace dnd;

int main() {
    ndarray a;
    cout << a << endl;
}

