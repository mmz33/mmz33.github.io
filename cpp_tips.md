---
layout: page
<!-- title: Effective Modern C++ -->
---

{% include mathjax.html %}

My language of choice is C++ because I believe that it is a really powerful programming language with a rich
set of concepts. I decided to write this blog post so that I can share my knowledge about how
to effectively use modern C++ and in particular C++11 and C++14. Not every C++ code is good and so
there are some ways that we need to follow to make sure we are benefiting from C++ new features. I am going to add add/update tips throughout my learning process.

## Tip 1: `auto` keyword

`auto` is a simple keyword that is used for type deduction. For example,

```cpp
auto x = 0.0;
auto s = "Hi";
```

Using `auto`, the compiler will deduce that x is a double variable and s is a string.
However, keep in mind that the initialization is important to deduce the type when using `auto`.

```cpp
int x;    // Ok but x is undefined
auto x;   // error! initializer is required
```

`auto` can helps in refactoring and also typing less. For example, if you want to write a code
that involves container's iterators such as for `std::vector`, then you can simple type the following:

```cpp
std::vector<int> v {1, 2, 3};

// Instead of std::vector<int>::iterator iter = v.begin()
auto iter = v.begin();
```

In addition, deducing types in `auto` is that same as template type deduction with one exception
which I am going to talk about it soon. Function templates will look like this:

```cpp
template<typename T>
void f(ParamType param);

f(expr); // call f with some expression
```

We can call then function `f` with some expression and the compiler will use this expr
to deduce the types of T and ParamType. Let's look at the following example:
We have this template:

```cpp
template<typename T>
void f(T& param);
```

These are the variable declarations:

```cpp
int x = 10;
const int cx = x;
const int& crx = x;
```

Then, the deduced types are as follows:

```cpp
f(x);   // T and ParamType types are both int
f(cx);  // T's type is const int, ParamType's type is const int&
f(crx); // T's type is const int, ParamType's type is const int&
```

Note that if param was passed as copy in the template function `f`, then T and ParamType will have the same types. Comparing this to `auto`, you can notice that `auto` plays the role of T in the template, and the type specifier for the variable acts as ParamType. As an example:

```cpp
// These are the same variables of the previous example
auto x = 10;
const auto cx = x;
const auto& crx = x;
```

However, there is one exception that occures when using `{}` initialization. `auto` keyword is able to deduce the type `std::initializer_list<T>` but templates can not do this implicitly. Let's make this more clear by an example:

```cpp
auto x = {1, 2, 3}; // x's type is std::initializer_list<int>

template<typename T>
void f(T x);

f({1, 2, 3}); // error! can not deduce the type for T
```

This is all for C++11, but for C++14 you can even use `auto` in functions and lambdas return types. However, it is important to know that auto will then be used as a template type deduction and not `auto` type deduction. Thus, the same rule applies for `std::initializer_list<T>` types.

```cpp
// C++14
auto f() {
  {1, 2, 3}; // error!
}

std::vector<int> v;
auto res = [&v](const auto& x) { v = x; };
res({1, 2, 3}); // error!
```

## Tip 2: Use `nullptr` instead of `0` and `NULL`

The problem here is that neither `0` nor `NULL` is of pointer type and so this can lead to ambiguity
sometimes. Lets consider this example:
```cpp
void f(int x);
void f(void*);

f(0); // calls f(int) and not f(void*)
f(NULL); // same behavoir as f(0)
f(nullptr); // calls f(void*)
```
We can see that the first two function calls are not really what we want. They are calling f with some kind
of integer type and not null pointer type whereas the third call with `nullptr` is the expected one. In addition, uncertainty
can also arise when using the keyword `auto` discussed in Tip 1.
```cpp
auto x = f(params);

if (x == 0) {
  ...
}
```
The problem in the above code is that we don't really know if the type of x is an integer or pointer. However, if we use `"if (x == nullptr) ..."` then it is obvious that the type of x is a pointer type. Therefore, to avoid ambiguity it is preferred to use `nullptr` to `0` and `NULL`.

## Tip 3: Difference between `()` and `{}`

This syntax confused me a lot. In C++11, you can have different syntax initialzation choices using paranthesis,
braces, or equal sign.
```cpp
// These all initialize x to 0 value

int x(0);
int x = 0;
int x{0};
int x = {0};
```
Too many options yeah? I think the most confusing one would be when you use `=` and you think it is an assignment
operation but it turns out it is calling the constructor. Here is an example:
```cpp
class A {
  ...
};

A a1; // calls default constructor
A a2 = a1; // this is not an assignment, it calls the copy constructor
a1 = a2; // this is an assignment
```
Regardless of having all these choices, *braces initialization* can be used almost anywhere but there are things that we need to be careful about which I am going to talk about them later. Now, lets focus on this initialization option. Using braces, it is easy to specify the elements of a container,
```cpp
std::vector<int> v {1, 2, 3}; // initialize a vector with elements 1, 2, 3
```
Moreover, one of the features of using braces initialization is that it prohobits implicity narrowing conversions among built-in types. In other words, compilers will complain if you are trying to express different types such as:
```cpp
double x, y, z;

int sum(x + y + z); // this is ok. It will be converted to int
int sum = x + y + z; // also ok
int sum{x + y + z}; // error! sum of doubles can't be expressed as int
```
Another advantage of using braces initialization is to avoid the confusion between declaring a function and
calling a constructor.
```cpp
A a1(5); // call A constructor with arg 5

// no args with parentheses means that you are declaring
// a function and not calling the constructor
A a2();

// calls A constructor with no args
A a3{};
```
Now, after talking about some of the features of brace initialzation, lets talk about some of it's drawbacks. Problems start to appear when using `std::initializer_list<T>` with braces initialzation and in particular when calling constructors. Lets begin with this healthy example:
```cpp
class A {
public:
  A(int x, bool b);
  A(int x, double d);
};

// calls first constructor
A a1(5, true);
A a2{5, true};

// calls second constructor
A a3(0, 10.0);
A a4{0, 10.0};
```
Till now everything is working so lets introduce `std::initializer_list<T>` to the game.
```cpp
class A {
public:
  A(int x, bool b);
  A(int x, double d);
  A(std::initializer_list<double> d);
};

A a1(5, true); // calls first constructor as before

// we expect that this will call the first constructor
// also but it will call the new constructor with
// std::initilaizer_list (5 and true are converted to double)
A a2{5, true};

A a3(0, 10.0); // calls second constructor as before

// same problem as a2
A a4{0, 10.0};
```
Oops... so now if we are using braces initialization instead of calling the constructor with the correct args
it is calling the constructor that is using `std::initializer_list`. This can really leads to ambiguity and
for sure wrong output.

I want to mention a rule here also that if you want to call the constructor with
empty `std::initializer_list` then you can't do it as `A a{}`. This will call the default constructor instead.
So what you can do is to use paranthesis as `A a({})`.

In the end, I just want to point out that we can easily have a design error because of braces initialization. A simple example would be when using the container `std::vector<T>`,
```cpp
std::vector<int> v1(10, 5); // creates 10-elements vector with 5 as value
std::vector<int> v2{10, 5}; // create 2-elements vector with values 10 and 5
```
You can see the problem above right? It is really confusing for the person that is using our code when we have such
design. This person must carefully choose between braces and paranthesis initialzation and so it is better
to design your constructors in a way that avoids as much problems as possible.

## Tip 4: Using Smart Pointers

Raw pointers are hard to love for many reasons such as we have to make sure that we destroyed a pointer only once with no resource leaks. Smart pointers come to avoid raw pointers issues. They are just wrappers around raw pointers that can do the same thing but with less error probability. The main smart pointers in C++11 are: `std::unique_ptr`, `std::shared_ptr`, and `std::weak_ptr`. Next I am going to give some tips how and when to use them effectively.

### 1. Use `std::unique_ptr` for exclusive-ownership resource management

`std::unique_ptr` is a fast and move-only pointer with exclusive ownership semantics. This means it owns what it is pointing to and thats why it is only movable. Moving a `std::unique_ptr` transfers ownership from the source pointer (set to nullptr) to the destination pointer. It can't be copied obviously because then two pointers would own the same object and that's against its semantics. As we said that such pointers are just wrappers of raw pointers and so upon destruction, `std::unique_ptr` will call the `destructor` by applying `delete` to the raw pointer. This happens when the `unique_ptr` goes out of scope. Lets consider the following class hierarchy:
```cpp
class Shape { ... };

class Circle : public shape { ... };
class Square : public shape { ... };
```
A common use for `std::unique_ptr` is as a factory function that allocates an object on the heap and returns a pointer to it, with the caller being responsible for deleting the object. This is a perfect match with the definition of `std::unique_ptr` because the caller will have an exclusive ownership about the object being allocated and it will automatically delete it when it is destroyed. A factory function for the above class hierarchy will look like the following:
```cpp
// return std::unique_ptr to an object created
// from the given args
template<typename... Ts>
std::unique_ptr<Shape> create_shape(Ts&&... params);
```
Caller then would use this function as:
```cpp
// caller scope
{
  ...
  auto shape_ptr = create_shape(params);
  ...
}
// destroy *shape_ptr (out of scope)
```
By default, the destruction is done via the `delete` keyword, however, you can create your own custom deleter if you want. This can be represented as a function such as function object or even lambda expressions. Suppose for example we want to do some logging before the destruction and so we need to create first a custom deleter.
```cpp
// custom deleter function
auto del_shape = [](Shape* shape) {
                  print_log(shape); // assume it is implemented
                  delete shape;
                 };

// factory function
template<typename... Ts>
std::unique_ptr<Shape, decltype(del_shape)>
create_shape(Ts&&... params) {
  std::unique_ptr<Shape, decltype(del_shape)>
    shape_ptr(nullptr, del_shape);

  if (...) {
    shape_ptr.reset(new Circle(std::forward<Ts>(params)...));
  } else if (...) {
    shape_ptr.reset(new Square(std::forward<Ts>(params)...));
  }
  return shape_ptr;
}
```
Lets try to understand now what is happening in the above code:
* del_shape is the custom deleter function that accepts a raw pointer to the object to be destroyed where in our case we do some logging first and then we delete the object.
* When using a custom deleter, it's type must be specified as in argument for `std::unique_ptr`. Thats why we added `decltype(del_shape)`
* In order to associate the custom deleter with shape_ptr, we need to add it to the constructor also.
* It is not possible to implicitly convert from raw pointer to `unique_ptr` and thats why we used the `reset` keyword when we wanted to create the required object. It give `shape_ptr` ownership on the object created via `new`.
* If you notice that the custom deleter `del_shape` takes as argument a raw pointer of type Shape and so regardless of the actual type created inside `create_shape` function (i.e circle or square) it will be deleted as a Shape object. This means that we will be deleting a derived class object via a base class pointer and so we need to have a virtual destructor for the base class:
```cpp
class Shape {
public:
    ...
    virtual ~Shape();
    ...
};
```
At the end, I just want to note about two things. First if you remember, I mentioned that we can also create the custom deleter via a function object instead of lambda expression but this will increase the size of the `std::unique_ptr` since we need to consider a function pointer now as an argument as: `std::unique_ptr<Shape, void (*)(Shape)>`.
Second, it is easy to convert from `std::unique_ptr` to `std::shared_ptr` (which will be explained next) by just doing the following: `std::shared_ptr<Shape> sp = create_shape(parmas)`
