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
        float v0[3] = {0.5, -1000, -2.2};
        int16_t v1[3] = {0, 0, 0};
        ndarray a = v0, b = v1;

        ndarray aview = a.as_dtype<double>();
        aview = aview.as_dtype<int32_t>(assign_error_none);
        //aview = aview.as_dtype<int16_t>(assign_error_none);

        ndarray bview = b.as_dtype<int32_t>(assign_error_none);
        //bview = bview.as_dtype<int64_t>(assign_error_none);


        cout << "aview: " << aview << endl;
        cout << "bview: " << bview << endl;
        bview.val_assign(aview, assign_error_none);
        cout << "aview: " << aview << endl;
        cout << "bview: " << bview << endl;
    } catch(std::exception& e) {
        cout << "Error: " << e.what() << "\n";
        return 1;
    }
}