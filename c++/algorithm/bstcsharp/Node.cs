using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace bstcsharp
{
    class Node
    {
        private int value;
        private Node right;
        private Node left;

        public Node(int value)
        {
            this.value = value;
            right = null;
            left  = null;
        }

        public void insert(Node node)
        {
            if (node.value < this.value)
            {
                if (this.left == null)
                    this.left = node;
                else
                    this.left.insert(node);
            }
            else if (node.value > this.value)
            {
                if (this.right == null)
                    this.right = node;
                else
                    this.right.insert(node);
            }
        }

        public bool search(int s)
        {
            if (this.value == s)
            {
                return true;
            }
            else if (s < this.value)
            {
                if (this.left == null)
                    return false;
                else
                    return this.left.search(s);
            }
            else if (s > this.value)
            {
                if (this.right == null)
                    return false;
                else
                    return this.right.search(s);
            }

            return false;
        }

        public void inorder() {
            if(this.left != null)
                this.left.inorder();

            Console.WriteLine(this.value);

            if(this.right != null)
                this.right.inorder();
        }
    }
}
