#include "ibex.h"
#include <iostream>
#include <chrono>

using namespace ibex;
using namespace std;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

NumConstraint func(const Array<const ExprSymbol> &args, const ExprCtr expr)
{
    cout << "size = " << args.size() << endl;
    switch (args.size())
    {
    case 1:
        return NumConstraint(args[0], expr);
    case 2:
        return NumConstraint(args[0], args[1], expr);
    case 3:
        return NumConstraint(args[0], args[1], args[2], expr);
    case 4:
        return NumConstraint(args[0], args[1], args[2], args[3], expr);
    case 5:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], expr);
    case 6:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], expr);
    case 7:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], expr);
    case 8:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], expr);
    case 9:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], expr);
    case 10:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], expr);
    case 11:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], expr);
    case 12:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], expr);
    case 13:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], expr);
    case 14:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], expr);
    case 15:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], expr);
    case 16:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], expr);
    case 17:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], expr);
    case 18:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], expr);
    case 19:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], expr);
    case 20:
        return NumConstraint(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], expr);
    default:
        return NumConstraint(args, expr);
    }
}

// void c1()
// {
//     int num_vars = 20;
//     Variable x[num_vars];

//     Array<const ExprSymbol> vars(num_vars);

//     const ExprNode *expr = &(x[0] + 1);
//     for (int i = 1; i < num_vars; i++)
//         expr = &(*expr + x[i]);
//     // expr = &(*expr < 0);

//     for (int i = 0; i < num_vars; i++)
//     {
//         vars.set_ref(i, x[i]);
//         // expr = expr + Function(x[i], x[i]);
//     }
//     // Function f(vars, *expr, "f");

//     NumConstraint c1 = func(vars, (*expr < 0)); // x[0] + x[1] + x[2] + x[3] + x[4] = 0);
//     cout << "Made constraint c1: " << c1 << endl;
// }

// void c2()
// {
//     int num_vars = 20;

//     // Function f(vars, *expr, "f");

//     NumConstraint c2(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], *expr < 0); // x[0] + x[1] + x[2] + x[3] + x[4] = 0);
//     cout << "Made constraint c2: " << c2 << endl;
// }

duration<double, std::milli> make_constraint(const ExprCtr e, Array<const ExprSymbol> &vars, bool use_opt)
{
    auto t1 = high_resolution_clock::now();
    // cout << vars.size() << endl;
    cout << "make_constraint(" << use_opt << ") - start" << endl;

    // cout << "expr = " << e << endl;
    NumConstraint c1 = use_opt ? func(vars, e) : NumConstraint(vars, e);

    cout << "make_constraint(" << use_opt << ") - end" << endl;
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    return ms_double;
}

const ExprNode *make_expr(Array<const ExprSymbol> &x)
{

    int num_vars = x.size();
    const ExprNode *expr = &(x[0] + 1);
    for (int i = 1; i < num_vars; i++)
        expr = &(*expr + x[i]);
    return expr;
}

void make_vars(Array<const ExprSymbol> *vars)
{
    int num_vars = vars->size();
    Variable x[num_vars];
    cout << "num_vars: " << num_vars << endl;

    for (int i = 0; i < num_vars; i++)
    {
        cout << "i = " << i << endl;
        vars->set_ref(i, x[i]);
    }
}

void summary(int num_vars, double time_with, double time_without)
{
    double speedup = time_without / time_with;
    cout << num_vars << ": " << time_with << " " << time_without << " (" << speedup << "x)" << endl;
}

int main()
{
    // Variable x;
    // NumConstraint c(x, x + 1 <= 0);
    // cout << "Made constraint: " << c << endl;
    cout << "Starting: " << endl;
    int num_vars = 10;
    Variable x[num_vars];
    Array<const ExprSymbol> vars(num_vars);

    for (int i = 0; i < num_vars; i++)
    {
        // cout << "i = " << i << endl;
        vars.set_ref(i, x[i]);
    }

    // const ExprNode *expr = &(vars[0] + 1);
    // for (int i = 1; i < num_vars; i++)
    //     expr = &(*expr + vars[i]);
    // const ExprCtr &e = (*expr < 0);

    // const ExprNode expr = ;
    const ExprCtr e = (vars[0] + vars[1] + vars[2] + vars[3] + vars[4] + vars[5] + vars[6] + vars[7] + vars[8] + vars[9]) < 0;
    auto time_without = make_constraint(e, vars, false);
    auto time_with = make_constraint(e, vars, true);
    // cout << "Made e" << endl;
    // const ExprNode *expr1 = &(vars[0] + 1);
    // for (int i = 1; i < num_vars; i++)
    //     expr1 = &(*expr1 + vars[i]);
    // cout << (*expr1) << endl;
    // cout << endl;

    // const ExprNode *expr1 = make_expr(vars);
    // cout << "Made expr1" << endl;
    // cout << (*expr) << endl;
    // const ExprCtr &e1 = (*expr < 0);

    summary(vars.size(), time_with.count(), time_without.count());
    // delete expr;

    // auto t1 = high_resolution_clock::now();
    // c1();
    // auto t2 = high_resolution_clock::now();
    // // c2();

    // auto t3 = high_resolution_clock::now();

    // /* Getting number of milliseconds as an integer. */
    // auto ms_int = duration_cast<milliseconds>(t2 - t1);

    // /* Getting number of milliseconds as a double. */
    // duration<double, std::milli> ms_double = t2 - t1;

    // std::cout << ms_int.count() << "ms\n";
    // std::cout << ms_double.count() << "ms\n";

    // /* Getting number of milliseconds as an integer. */
    // auto ms_int1 = duration_cast<milliseconds>(t3 - t2);

    // /* Getting number of milliseconds as a double. */
    // duration<double, std::milli> ms_double1 = t3 - t2;

    // // std::cout << ms_int1.count() << "ms\n";
    // // std::cout << ms_double1.count() << "ms\n";

    return 0;
}
