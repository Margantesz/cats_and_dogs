def load_Xy(path):
    filenames0 = glob(path + "norm/*.png")
    filenames1 = glob(path + "trypo/*.png")
    X = np.zeros((len(filenames0) + len(filenames1), 256, 256, 3))

    y = np.zeros(len(filenames0) + len(filenames1))
    y[len(filenames0):] = 1.

    print(path + "norm/*.png")
    for i, filename in enumerate(filenames0):
        X[i] = plt.imread(filename)
        if i % 100 == 0:
            print(i, end=" ")
        if i % 1000 == 0:
            print("")

    print("\n")
    print(path + "trypo/*.png")
    for i, filename in enumerate(filenames1):
        X[len(filenames0) + i] = plt.imread(filename)
        if i % 100 == 0:
            print(i, end=" ")
        if i % 1000 == 0:
            print("")

    return X, y


