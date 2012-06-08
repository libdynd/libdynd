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
    float v0[5] = {3.5f, 1.3f, -2.4999f, -2.999, 1000.50001f};
    ndarray a = v0, b;

    b = a.as_dtype<int>(assign_error_overflow);
    b = b.as_dtype<float>(assign_error_inexact);
    // Multiple as_dtype operations should make a chained conversion dtype

    // Evaluating the values should truncate them to integers
    cout << b << endl;
    b = b.vals();
    cout << b << endl;

    // Now try it with longer chaining through multiple item sizes
    b = a.as_dtype<int16_t>(assign_error_overflow);
    b = b.as_dtype<int32_t>(assign_error_overflow);
    b = b.as_dtype<int16_t>(assign_error_overflow);
    b = b.as_dtype<int64_t>(assign_error_overflow);
    b = b.as_dtype<float>(assign_error_overflow);
    b = b.as_dtype<int32_t>(assign_error_overflow);

    cout << b << endl;
    b = b.vals();
    cout << b << endl;
}