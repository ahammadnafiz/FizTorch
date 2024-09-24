from ftensor import FTensor

def main():
    a = FTensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = FTensor([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])
    result = a.dot(b)
    print(a + b)
    print(a - b)
    print(a * b)
    print(result)
    print(a.flatten())
    print(b.transpose())
    c = FTensor([[1, 2, 4]])
    print(c.shape)
    print(c.transpose().shape)
    d = FTensor([[1, 2, 3], [4, 5, 6]])
    print(d.sum(axis=0))
    print(d.sum(axis=1))
    print(d.sum())
    print(b.size)

if __name__ == "__main__":
    main()