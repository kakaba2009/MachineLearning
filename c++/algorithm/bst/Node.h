#pragma once

using namespace std;

class Node
{
public:
	Node();

	Node(int value);

	~Node();

	Node* addNode(Node* root);

	int getValue() const;

	void inOrder() const;

	friend ostream& operator<<(ostream& os, const Node& node);

private:
	int  value;
	Node *left;
	Node *right;
};

class BSTree
{
public:
	BSTree() {
	}

	~BSTree() {
	}

	Node* addNode(int value) {
		//cout << value << endl;

		Node* node = new Node(value);

		if (root == nullptr) {
			return(root = node);
		}

		return root->addNode(node);
	}

	void inOrder() {
		root->inOrder();
	}

private:
	Node* root = nullptr;
};

