#include <iostream>

using namespace std;

struct Node
{
    int data;
    Node* next;

    Node(int val)
    {
        data = val;
        next = NULL;
    }
};

int size(Node* &head)
{
    int size = 0;
    Node* n = head;
    while(n!=NULL)
    {
        n = n->next;
        size++;
    }
    return(size);
}

void insertAtStart(Node* &head, int x)
{
    Node* new_node = new Node(x);
    new_node->next = head;
    head = new_node;
}

void insertAtTail(Node* &head, int x)
{
    Node* n = new Node(x);
    
    if(head == NULL)
    {
        head = n;
        return;
    }

    Node* temp = head;
    while(temp->next!=NULL)
   {
        temp=temp->next;
   }

   temp->next = n; 
    
}

void insert(Node* &head, int x, int k)   //inserting a node with data x at position k where k = 0, 1, 2, 3.... Here if k> size of linked list, it inserts element at the last
{
    Node* n =new Node(x);
    
    if(k==0)
    {
        insertAtStart(head, x);
        return;
    }

    Node* prev = head;
    Node* current;

    while(--k)
    {
        if(prev->next == NULL)
        {
            insertAtTail(head,x);
            return;
        }
        prev = prev->next;
    }

    current = prev->next;
    prev->next = n;
    n->next = current;
}

void deleteNode(Node* &head, int k)      //deleting a node at the kth position where k = 0, 1, 2, 3...
{
    if(k==0)
    {
        head = head->next;
        return;
    }

    Node* prev = head;
    Node* current;

    while(--k)
    {
        prev = prev->next;
        if (prev==NULL)
        {
            cout<<"There is no node at this position\n";
            return;
        }
    }

    current = prev->next->next;
    prev->next = current;
}

Node* reverse(Node* n)
{
    Node* current = n;
    Node* prev = NULL;
    Node* temp;
    while(current!=NULL)
    {
        temp = current->next;
        current->next = prev;
        prev = current;
        current = temp;
    }

    return prev;
}

void reverseUsingRecursion(Node* n, Node* &head)
{
    if(n->next == NULL)
    {
        head = n;
        return;
    }

    reverseUsingRecursion(n->next, head);
    Node* q = n->next;
    q->next = n;
    n->next = NULL;
}

void reverseSetOfNodes(Node* &head, int start, int end)      //reversing set of nodes from positons start to end, both inclusive
{
    if(start>end)
    {
        cout<<"Invlid parameters\n";
        return;
    }
    Node* s = head;
    Node* e = head;
    Node* prev = NULL;
    for(int i = 0; i<end; i++)
    {
        if(i<start)
        {
            prev = s;
            s = s->next;
        }
        if(e->next == NULL)
        break;
        e = e->next;
    }
    Node* tail = e->next;
    e->next = NULL;     //by making e->next = NULL, we create a independent linked list from start to end. So now pointer of start can be sent to reverse() to reverse the linked list from start to end
    if(start==0)
    {
        e = reverse(s);
        head = e;
        s->next = tail;
        return;
    }
    e = reverse(s);
    prev->next = e;
    s->next = tail;    
}

void reversek(Node* &head, int k)       //reverses every k nodes
{
    int count;
    if(size(head)%k==0)
    count = size(head)/k;
    else
    count = (size(head)/k)+1;
    int start = 0, end = k-1;
    for(int i=0;i<count;i++)
    {
        reverseSetOfNodes(head,start,end);
        start+=k;
        end+=k;
    }
}

void printList(Node* n)
{
    while(n!=NULL)
   {
        cout<<n->data<<" ";
        n=n->next;
   } 
   cout<<"\n";
}

void printReverse(Node* n)
{
    if(n == NULL)
    return;

    printReverse(n->next);
    cout<<n->data<<" ";
}

int main()
{
    Node* head = NULL;
    Node* second = NULL;
    Node* third = NULL;

    head = new Node(1);
    second = new Node(2);
    third = new Node(5);
    
    head->next=second;
    second->next=third;
    third->next=NULL;

    insertAtTail(head, 10);
    insertAtStart(head, 23);
    insertAtStart(head, 54);
    insertAtStart(head, 14);
    insertAtTail(head,23);
    insert(head, 15, 2);
    insert(head, 7, 0);
    insert(head, 85, 100);
    head = reverse(head);
    deleteNode(head, 2);
    deleteNode(head, 0);
    reverseUsingRecursion(head, head);
    printList(head);
    reverseSetOfNodes(head,2,5);
    printList(head);
    reversek(head,7);
    printList(head);
}