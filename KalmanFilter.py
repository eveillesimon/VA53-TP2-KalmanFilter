import numpy as np
from sympy import pprint

class KalmanFilter():
    def __init__(self, A: np.ndarray, B: np.ndarray, H: np.ndarray, R: np.ndarray, Q: np.ndarray):
        self.A = A
        self.B = B
        self.H = H
        self.R = R
        self.Q = Q
        self.Uk = np.ones((self.A.shape[0], 1))
        print("Uk")
        pprint(self.Uk)
        self.Xprec_post = None
        self.Pprec_post = None
        self.Xcurr_prior = None
        self.Pcurr_prior = None

    def initialize(self, X0, P0):
        self.Xprec_post = X0
        self.Pprec_post = P0

    def predict(self):
        assert self.Xprec_post is not None, "You must initialize a first estimation of the state"
        assert self.Pprec_post is not None, "You must initialize a first error on covariance"

        self.Xcurr_prior = (self.A @ self.Xprec_post) + (self.B @ self.Uk)
        self.Pcurr_prior = (self.A @ self.Pprec_post @ self.A.T) + self.Q

        return self.Xcurr_prior

    def correct(self, Zmeasure):
        assert self.Xcurr_prior is not None, "You must predict before correct !"
        assert self.Pcurr_prior is not None, "You must predict before correct !"

        self.Kk = self.Pcurr_prior @ self.H.T @ np.linalg.inv((self.H @ self.Pcurr_prior @ self.H.T) +  self.R)
        self.Xcurr_post = self.Xprec_post + (self.Kk @ (Zmeasure - (self.H @ self.Xcurr_prior)))
        self.Pcurr_post = (np.identity(self.Kk.shape[0]) - (self.Kk @ self.H)) @ self.Pcurr_prior

        self.Xprec_post = self.Xcurr_post
        self.Pprec_post = self.Pprec_post

        return self.Xcurr_post



class WalkingPedestrianKalmanFilter(KalmanFilter):
    def __init__(self, dt, rx, ry, q):

        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        B = np.zeros((A.shape[0], A.shape[1]))

        H = np.concatenate((np.identity(2), np.zeros((2,2))), axis=1)
        print(H)

        R = np.array([
            [rx, 0],
            [0, ry]
        ])

        Q = q * np.identity(A.shape[0])

        super().__init__(A, B, H, R, Q)


    def initialize(self, x0, y0, vx0, vy0, p):

        X0 = np.array([
            [x0],
            [y0],
            [vx0],
            [vy0]
        ])

        P0 = p * np.identity(self.A.shape[0])

        super().initialize(X0, P0)