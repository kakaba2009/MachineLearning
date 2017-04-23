using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace bstcsharp
{
    class BSTree
    {
        private Node root;

        public BSTree() {
            root = null;
        }

        public void insert(int d)
        {
            Node node = new Node(d);

            if (root == null)
            {
                root = node;
            }

            root.insert(node);
        }

        public bool search(int s) {
            bool result = root.search(s);

            if(result)
                Console.WriteLine("Found {0}", s);
            else
                Console.WriteLine("Not Found {0}", s);

            return result;
        }

        public void inorder() {
            if (root != null)
                root.inorder();
        }
    }
}