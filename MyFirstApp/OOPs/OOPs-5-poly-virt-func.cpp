#include<bits/stdc++.h>
using namespace std;

class base{
    public:
    virtual void print(){
        cout<<"I am print function in base class"<<endl;
    }
    virtual void get(){
        cout<<"I am get function in base class"<<endl;
    }
};

class derived : public base{
    public:
     void print(){
        cout<<"I am print function in derived class"<<endl;
    }
     void get(){
        cout<<"I am get function in derived class"<<endl;
    }
};

int main(){
    base *baseptr;
    derived d;
    baseptr=&d;
    baseptr -> print();
    baseptr -> get();
return 0;
}