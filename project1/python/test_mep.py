from matrixelementparser import MatrixElementParser
def test_mep():
    matrix = MatrixElementParser(Z=2)

    assert matrix[0,1,0,1] == -matrix[0,1,1,0] == 1.25
    p = 1
    q = 2
    r = 1
    s = 2
    print([k//2 + 1 for k in (p,q,r,s)])
    Z = 2
    print((17*Z)/81)
    print(matrix[p,q,r,s])
    print(matrix[q,p,s,r])

if __name__ == "__main__":
    test_mep()
