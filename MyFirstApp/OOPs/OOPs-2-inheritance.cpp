#include<iostream>
#include<bits/stdc++.h>
using namespace std;

class A{
    public:
    void func(){
        cout<<"Inherited"<<endl;
    }
};

class C{
    public:
    void funcC(){
        cout<<"Inherited 2"<<endl;
    }
};

class B : public A,public C{
    public:
};

int main(){
    B b;
    b.func();
    b.funcC();
    
return 0;
}