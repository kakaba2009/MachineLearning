using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace bstcsharp
{
    class Program
    {
        static void Main(string[] args)
        {
            BSTree b = new BSTree();

            b.insert(3);
            b.insert(6);
            b.insert(2);
            b.insert(9);
            b.insert(5);
            b.insert(1);

            b.inorder();

            b.search(5);
            b.search(100);

            Console.ReadLine();
        }
    }
}
