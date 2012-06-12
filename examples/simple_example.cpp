#include <iostream>

#include <dnd/ndarray.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>

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

#define EXPECT_EQ(a, b) \
    cout << "first   : " << (a) << endl \
         << "second  : " << (b) << endl

int main()
{
    try {
        ndarray a;

        // Basic case with no buffering
        a = ndarray(2) * ndarray(3);
        EXPECT_EQ(elementwise_binary_op_node_type, a.get_expr_tree()->get_node_type());
        EXPECT_EQ(make_dtype<int>(), a.get_expr_tree()->get_dtype());
        EXPECT_EQ(make_dtype<int>(), a.get_expr_tree()->get_opnode(0)->get_dtype());
        EXPECT_EQ(make_dtype<int>(), a.get_expr_tree()->get_opnode(1)->get_dtype());
        EXPECT_EQ(6, a.as<int>());

        // Buffering the first operand
        a = ndarray(2) * ndarray(3.f);
        EXPECT_EQ(elementwise_binary_op_node_type, a.get_expr_tree()->get_node_type());
        EXPECT_EQ(make_dtype<float>(), a.get_expr_tree()->get_dtype());
        EXPECT_EQ((make_conversion_dtype<float, int>()), a.get_expr_tree()->get_opnode(0)->get_dtype());
        EXPECT_EQ(make_dtype<float>(), a.get_expr_tree()->get_opnode(1)->get_dtype());
        EXPECT_EQ(6, a.as<float>());

        // Buffering the second operand
        a = ndarray(2.f) * ndarray(3);
        EXPECT_EQ(elementwise_binary_op_node_type, a.get_expr_tree()->get_node_type());
        EXPECT_EQ(make_dtype<float>(), a.get_expr_tree()->get_dtype());
        EXPECT_EQ(make_dtype<float>(), a.get_expr_tree()->get_opnode(0)->get_dtype());
        EXPECT_EQ((make_conversion_dtype<float, int>()), a.get_expr_tree()->get_opnode(1)->get_dtype());
        EXPECT_EQ(6, a.as<float>());

        // Buffering the output
        a = (ndarray(2) * ndarray(3)).as_dtype<float>();
        EXPECT_EQ(elementwise_binary_op_node_type, a.get_expr_tree()->get_node_type());
        EXPECT_EQ((make_conversion_dtype<float, int>()), a.get_expr_tree()->get_dtype());
        EXPECT_EQ(make_dtype<int>(), a.get_expr_tree()->get_opnode(0)->get_dtype());
        EXPECT_EQ(make_dtype<int>(), a.get_expr_tree()->get_opnode(1)->get_dtype());
        EXPECT_EQ(6, a.as<float>());

        // Buffering both operands
        a = ndarray(2) * ndarray(3u).as_dtype<float>();
        EXPECT_EQ(elementwise_binary_op_node_type, a.get_expr_tree()->get_node_type());
        EXPECT_EQ(make_dtype<float>(), a.get_expr_tree()->get_dtype());
        EXPECT_EQ((make_conversion_dtype<float, int>()), a.get_expr_tree()->get_opnode(0)->get_dtype());
        EXPECT_EQ((make_conversion_dtype<float, unsigned int>()), a.get_expr_tree()->get_opnode(1)->get_dtype());
        EXPECT_EQ(6, a.as<float>());

        // Buffering the first operand and the output
        a = (ndarray(2) * ndarray(3.f)).as_dtype<double>();
        EXPECT_EQ(elementwise_binary_op_node_type, a.get_expr_tree()->get_node_type());
        EXPECT_EQ((make_conversion_dtype<double, float>()), a.get_expr_tree()->get_dtype());
        EXPECT_EQ((make_conversion_dtype<float, int>()), a.get_expr_tree()->get_opnode(0)->get_dtype());
        EXPECT_EQ(make_dtype<float>(), a.get_expr_tree()->get_opnode(1)->get_dtype());
        EXPECT_EQ(6, a.as<double>());

        // Buffering the second operand and the output
        a = (ndarray(2.f) * ndarray(3)).as_dtype<double>();
        EXPECT_EQ(elementwise_binary_op_node_type, a.get_expr_tree()->get_node_type());
        EXPECT_EQ((make_conversion_dtype<double, float>()), a.get_expr_tree()->get_dtype());
        EXPECT_EQ(make_dtype<float>(), a.get_expr_tree()->get_opnode(0)->get_dtype());
        EXPECT_EQ((make_conversion_dtype<float, int>()), a.get_expr_tree()->get_opnode(1)->get_dtype());
        EXPECT_EQ(6, a.as<double>());

        // Buffering both operands and the output
        a = (ndarray(2) * ndarray(3u).as_dtype<float>()).as_dtype<double>();
        EXPECT_EQ(elementwise_binary_op_node_type, a.get_expr_tree()->get_node_type());
        EXPECT_EQ((make_conversion_dtype<double, float>()), a.get_expr_tree()->get_dtype());
        EXPECT_EQ((make_conversion_dtype<float, int>()), a.get_expr_tree()->get_opnode(0)->get_dtype());
        EXPECT_EQ((make_conversion_dtype<float, unsigned int>()), a.get_expr_tree()->get_opnode(1)->get_dtype());
        EXPECT_EQ(6, a.as<double>());

    } catch(int){//std::exception& e) { cout << "Error: " << e.what() << "\n";
        return 1;
    }
}