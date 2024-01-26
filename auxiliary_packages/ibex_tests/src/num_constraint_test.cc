#include "ibex.h"
#include <iostream>
#include <chrono>

using namespace ibex;
using namespace std;

NumConstraint func(Array<ExprSymbol> &args, const ExprCtr &expr)
{
    return NumConstraint(args, &expr);
}

void c1()
{
    using namespace std::chrono_literals;
    int num_vars = 20;
    Variable x[num_vars];

    Array<const ExprSymbol> vars(num_vars);

    const ExprNode *expr = &(x[0] + 1);
    for (int i = 1; i < num_vars; i++)
        expr = &(*expr + x[i]);
    // expr = &(*expr < 0);

    for (int i = 0; i < num_vars; i++)
    {
        vars.set_ref(i, x[i]);
        // expr = expr + Function(x[i], x[i]);
    }
    // Function f(vars, *expr, "f");

    NumConstraint c1(func(&vars, *expr < 0)); // x[0] + x[1] + x[2] + x[3] + x[4] = 0);
    cout << "Made constraint c1: " << c1 << endl;
}

void c2()
{
    using namespace std::chrono_literals;
    int num_vars = 20;
    Variable x[num_vars];

    Array<const ExprSymbol> vars(num_vars);
    const ExprNode *expr = &(x[0] + 1);
    for (int i = 1; i < num_vars; i++)
        expr = &(*expr + x[i]);
    // expr = &(*expr < 0);

    for (int i = 0; i < num_vars; i++)
    {
        vars.set_ref(i, x[i]);
        // expr = expr + Function(x[i], x[i]);
    }
    // Function f(vars, *expr, "f");

    NumConstraint c2(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], *expr < 0); // x[0] + x[1] + x[2] + x[3] + x[4] = 0);
    cout << "Made constraint c2: " << c2 << endl;
}

int main()
{
    Variable x;
    NumConstraint c(x, x + 1 <= 0);
    cout << "Made constraint: " << c << endl;

    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    c1();
    auto t2 = high_resolution_clock::now();
    c2();

    auto t3 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";

    /* Getting number of milliseconds as an integer. */
    auto ms_int1 = duration_cast<milliseconds>(t3 - t2);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double1 = t3 - t2;

    std::cout << ms_int1.count() << "ms\n";
    std::cout << ms_double1.count() << "ms\n";

    return 0;
}
