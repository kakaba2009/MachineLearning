

class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

    def insert(self, node):
        if node.value < self.value:
            if self.left is None:
                self.left = node
            else:
                self.left.insert(node)
        elif node.value > self.value:
            if self.right is None:
                self.right = node
            else:
                self.right.insert(node)

    def in_order(self):
        if self.left is not None:
            self.left.in_order()

        print(self.value)

        if self.right is not None:
            self.right.in_order()


class BSTree:
    def __init__(self):
        self.root = None

    def add_node(self, value):
        node = Node(value)

        if self.root is None:
            self.root = node

        self.root.insert(node)

    def in_order(self):
        self.root.in_order()

root = BSTree()

root.add_node(5)
root.add_node(6)
root.add_node(1)
root.add_node(8)
root.add_node(10)
root.add_node(7)

root.in_order()
