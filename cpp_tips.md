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
