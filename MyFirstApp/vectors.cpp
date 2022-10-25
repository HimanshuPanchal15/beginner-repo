#include<iostream>
#include<vector>
using namespace std;

template <class T>   // creating a template class so that any type of function can be called not necessarily int.
void display(vector<T> &v)
{
    for(int i=0;i<v.size();i++)  // size function
    {
        cout<<v[i]<<" ";
        // cout<<v.at(i)<<" ";  // at function
    }
}
int main(){
    // Creating a vector
    vector<int> vec1;   // zero length integer vector
    // int size,e;
    // cout<<"Enter the size: ";
    // cin>>size;
    // for(int i=0;i<size;i++)
    // {
    //     cout<<"Add elements to this vector: ";
    //     cin>>e;
    //     vec1.push_back(e);  //  push_back function
    // }
    // vec1.pop_back();  // pop_back function
    // display(vec1);
    // cout<<endl;

    // vector<int> ::iterator iter=vec1.begin();  // begin function
    // vec1.insert(iter+1,6,5);   // insert function

    display(vec1);
    cout<<endl;

    vector<char> vec2(4);// 4 length char vector
    vec2.push_back('5');
    display(vec2);
    cout<<endl;

    vector<char> vec3(vec2); // 4-length char vector from vec2.
    display(vec3);
    cout<<endl;

    vector<int> vec4(6,13); // 6-element vector of 13s
    display(vec4);
    cout<<endl;

return 0;
}