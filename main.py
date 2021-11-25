import matplotlib.pyplot as plt
import numpy as np

V_top = 100
V_bottom = 0
V_left = 20
V_right = 80

iteration = 500


class Laplace:
    def __init__(self, gridsize, h, q, iteration):

        self.gridsize = gridsize
        self.h = h
        self.rho_epsilon = q
        self.iteration = iteration

    def Meshgrid(self):
        X, Y = np.meshgrid(np.arange(self.gridsize), np.arange(self.gridsize))
        return X, Y

    def Dirichlet_BC(self, X, Y):
        V = np.zeros((len(X), len(Y)))

        V[: (len(Y) - 1), :] = V_bottom
        V[:1, :] = V_top
        V[:, :1] = V_left
        V[:, (len(X) - 1) :] = V_right

        return V

    def jacobi(self, V, i, j):
        return (
            1
            / 4
            * (
                V[j + 1, i]
                + V[j - 1, i]
                + V[j, i - 1]
                + V[j, i + 1]
                + self.h ** 2 * self.rho_epsilon
            )
        )

    def iterative(self, V, X, Y):
        h = self.h
        error = 0
        maximumError = 0.0001

        Ex, Ey = np.zeros(len(X)), np.zeros(len(Y))

        for n in range(self.iteration):
            for i in range(1, len(X) - 1, h):
                for j in range(1, len(Y) - 1, h):
                    v = self.jacobi(V, i, j)
                    dv = V[j, i] - v
                    V[j, i] = v
                    Ey[j] = -(V[j + 1, i] - V[j - 1, i]) / (2 * h)

                    error = max(error, dv)
                Ex[i] = -(V[j, i + 1] - V[j, i - 1]) / (2 * h)

        if error < maximumError:
            print("Computation done!")
        else:
            print("Iteration times is not high enough.")

        return Ex, Ey

    def Plot_contour(self, X, Y, V):
        colorinterpolation = 100
        colourMap = plt.cm.jet

        plt.title("Contour of Electric Potential")
        plt.contourf(X, Y, V, colorinterpolation, cmap=colourMap)

        plt.colorbar()
        plt.savefig("Electric_Potential_contour.png")
        plt.show()

    def Plot_vf(self, X, Y, Ex, Ey):
        plt.title("Electric Vector Field")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.quiver(X, Y, Ex * X, Ey * Y)
        plt.savefig("Electric_field.png")
        plt.show()

    def Normal(self, X, Y):
        path = np.zeros((len(X), len(Y)))

        path[:2, :] = 1
        path[len(Y) - 2 :, :] = -1
        path[:, :2] = -1
        path[:, len(X) - 2 :] = +1

        path[:, :1] = 0
        path[:, (len(X) - 1) :] = 0
        path[:1, :] = 0
        path[(len(Y) - 1) :, :] = 0

        return path

    def Calculate(self, Ex, Ey, X, Y):
        N = self.Normal(X, Y)

        xx = np.dot(Ex * X, N)
        yy = np.dot(Ey * Y, N)

        summ = 0

        for j in range(len(Y)):
            for i in range(len(X)):
                summ += np.dot(xx, yy)[j, i]

        return summ


def main():
    sheet = Laplace(100, 1, 5, 500)
    X, Y = sheet.Meshgrid()
    V = sheet.Dirichlet_BC(X, Y)
    Ex, Ey = sheet.iterative(V, X, Y)
    sheet.Plot_contour(X, Y, V)
    sheet.Plot_vf(X, Y, Ex, Ey)
    summ = sheet.Calculate(Ex, Ey, X, Y)
    print(summ)


main()
