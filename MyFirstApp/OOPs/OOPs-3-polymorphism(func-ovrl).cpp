#include<iostream>
#include<bits/stdc++.h>
using namespace std;

class A{
    public:
    void fun(){
        cout<<"Function with no argument"<<endl;
    }
    void fun(int){
        cout<<"Function with int argument"<<endl;
    }
    void fun(double){
        cout<<"Function with double argument"<<endl;
    }
};

int main(){
    A a;
    a.fun();
    a.fun(4);
    a.fun(4.23);
    
return 0;
}