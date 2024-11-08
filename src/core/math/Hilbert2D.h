#include <stdio.h>

namespace Flair
{
    class Hilbert2D
    {
    private:
        static void Rot(int dim, int& x, int& y, int rx, int ry)
        {
            if (ry == 0)
            {
                if (rx == 1)
                {
                    x = dim - 1 - x;
                    y = dim - 1 - y;
                }

                int t = x;
                x = y;
                y = t;
            }
        }

    public:
        Hilbert2D() = delete;

        // Maps (x,y) coordinates in the pow2 square of size dim to the Hilbert curve parameter t
        static int ToCurve(int dim, int x, int y)
        {
            int rx, ry, s, d = 0;
            for (s = dim / 2; s > 0; s /= 2)
            {
                rx = (x & s) > 0;
                ry = (y & s) > 0;
                d += s * s * ((3 * rx) ^ ry);
                Rot(dim, x, y, rx, ry);
            }
            return d;
        }

        // Maps the Hilbert curve parameter t to coordinates (x, y) on pow2 square of size dim to 
        static void FromCurve(int dim, int t, int& x, int& y)
        {
            int rx, ry, s;
            x = y = 0;
            for (s = 1; s < dim; s *= 2)
            {
                rx = 1 & (t / 2);
                ry = 1 & (t ^ rx);
                Rot(s, x, y, rx, ry);
                x += s * rx;
                y += s * ry;
                t /= 4;
            }
        }
    };
}