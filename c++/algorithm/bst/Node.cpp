#include <iostream>
#include "Node.h"

using namespace std;

Node::Node() : value(0), left(nullptr), right(nullptr) {
}

Node::Node(int val) : value(val), left(nullptr), right(nullptr) {
}

Node::~Node(){
	delete left;
	delete right;
}

Node* Node::addNode(Node* node) {
	if (node->getValue() < value) {
		if (this->left == nullptr) {
			return(this->left = node);
		}
		else {
			this->left->addNode(node);
		}
	}
	else if (node->getValue() > value) {
		if(this->right == nullptr) {
			return(this->right = node);
		}
		else {
			this->right->addNode(node);
		}
	}
}

int Node::getValue() const {
	return value;
}

void Node::inOrder() const {
	if (this->left != nullptr) {
		this->left->inOrder();
	}

	cout << value << " ";

	if (this->right != nullptr) {
		this->right->inOrder();
	}
}

ostream& operator<<(ostream& os, const Node& node) {
	cout << node.getValue();
	return os;
}

int main() {
	BSTree* root = new BSTree();

	root->addNode(5);
	root->addNode(6);
	root->addNode(1);
	root->addNode(8);
	root->addNode(10);
	root->addNode(7);

	root->inOrder();

	char w;

	cin >> w;
}