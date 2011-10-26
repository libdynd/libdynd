#include <dnd/ndarray.hpp>

using namespace std;
using namespace dnd;

int main()
{
    ndarray a, b, c;

#ifdef DND_INIT_LIST
    // C++11 Initializer lists!
    a = {{3,2,1}, {4,5,6}};
    b = {1,-1,1};
#else
    {int tmp[2][3] = {{3,2,1}, {4,5,6}}; a = tmp;}
    {int tmp[3] = {1,-1,1}; b = tmp;}
#endif

    c = a + b;

    cout << "a: " << a << "\n";
    cout << "b: " << b << "\n";
    cout << "c (a+b): " << c << "\n";

    // 'c' contains an expression tree
    cout << "\ndump of c:\n";
    c.debug_dump(cout);
    cout << "\n";

    // 'c' is a view of 'a+b', so changing 'a' changes 'c'
    a(1,0).vals() = 1000;
    cout << c << "\n";

    // This produces a compile error! TODO: Need tests to confirm stuff like this, too!
    //a(1,0) = 1000;
}
