#include<iostream>
#include<bits/stdc++.h>
using namespace std;

class student{
    string name;
public:  //data above this line is private
    
    int age;
    bool gender;

    student(){  
        cout<<"default constructor"<<endl;
    }

    student(string s,int a, bool g){   //parameterized constructor
        name=s;
        age=a;
        gender=g;
    }

    student(student &a){
        cout<<"copy constructor"<<endl;
        name=a.name;
        age=a.age;
        gender=a.gender;
    }

    ~student(){
        cout<<"destructor called"<<endl;
    }
    void setName(string s){
        name=s;
    }

    void getName(){
        cout<<name<<endl;
    }

    void printInfo(){  //can use such functions to access private data members
        cout<<"Name: ";
        cout<<name<<endl;
        cout<<"age: ";
        cout<<age<<endl;
        cout<<"Gender: ";
        cout<<gender<<endl;
    }

    bool operator == (student &a){  //operator overloading
        if(name==a.name && age==a.age && gender==a.gender){  //here name,age & gender are of c
            return true;
        }
        else 
            return false;
    }
};

int main(){
    // student a = student("a",11,1); //wont work if constructor specified
    // a.name="Piyush";
    // a.age=18;
    // a.gender=0;

    // student arr[3];
    // for(int i=0;i<3;i++)
    // {
    //     cout<<"Name: ";
    //     string s;
    //     cin>>s;
    //     arr[i].setName(s);
    //     cout<<"Age: ";
    //     cin>>arr[i].age;
    //     cout<<"Gender: ";
    //     cin>>arr[i].gender;
    // }
    // for(int i=0;i<3;i++)
    // {
    //     arr[i].printInfo();
    // }

    student a("Piyush",18,1);
    a.printInfo();
    student b("b",11,1);
    b.printInfo();

    student e;  //default constructor
    // cout<<e.age<<endl;
    student c=a; //copy constructor
    student d("d",11,1);
    if(d==a){  // wont work if operator not overloaded compared with the 2nd val
    // if(c==a){ 
        cout<<"same"<<endl;
    }
    else
        cout<<"not same"<<endl;

return 0;
}