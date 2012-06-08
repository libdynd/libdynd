#include <dnd/ndarray.hpp>

using namespace std;
using namespace dnd;

int main1()
{
    try {
        ndarray a;

        //a = {1, 5, 7};
        int avals[] = {1, 5, 7};
        a = avals;

        cout << a << endl;

        ndarray a2 = a.as_dtype<float>();

        cout << a2 << endl;

        ndarray a3 = a.as_dtype<double>();

        cout << a3 << endl;

        ndarray a4 = a2.as_dtype<double>();

        cout << a4 << endl;
        return 0;

        float avals2[2][3] = {{1,2,3}, {3,2,9}};
        ndarray b = avals2;

        ndarray c = a + b;

        c.debug_dump(cout);
        cout << c << endl;

        cout << c(0,1) << endl;
        a(1).val_assign(1.5f);
        cout << c(0,1) << endl;

        return 0;
    } catch(std::exception& e) {
        cout << "Error: " << e.what() << "\n";
        return 1;
    }
}

int main()
{
    try {
        float v0[3] = {0, 0, 0};
        ndarray a = v0, b;

        b = a.as_dtype<int16_t>(assign_error_inexact);
        b = b.as_dtype<double>(assign_error_overflow);
        // Multiple as_dtype operations should make a chained conversion dtype

        cout << "a: " << a << endl;
        cout << b << endl;
        b(0).vals() = 6.8f;
        cout << "a: " << a << endl;
        cout << b << endl;
        b(1).vals() = -3.1;
        cout << b << endl;
        b(2).vals() = 1000.5;
        cout << b << endl;

        cout << b.vals() << endl;
    } catch(std::exception& e) {
        cout << "Error: " << e.what() << "\n";
        return 1;
    }
}