import numpy as np

class Body_R():

    @staticmethod
    def R_AR(Marker1, Marker2, Marker3, Marker4, Marker5, Marker6):

        O_AR = (Marker2 + Marker5) * 0.5

        Y_AR = (Marker3 + Marker6) * 0.5 - O_AR
        Y_AR = Y_AR / np.linalg.norm(Y_AR)  # normalization

        A_AR = (Marker1 + Marker4) * 0.5 - O_AR

        X_AR = np.cross(Y_AR, A_AR)
        X_AR = X_AR / np.linalg.norm(X_AR)  # normalization

        Z_AR = np.cross(Y_AR, X_AR)

        R_AR = np.hstack([X_AR.reshape(-1, 1), Y_AR.reshape(-1, 1), Z_AR.reshape(-1, 1)])

        return O_AR, X_AR, Y_AR, Z_AR, R_AR
    
    @staticmethod
    def Rs_AR(Markers1, Markers2, Markers3, Markers4, Markers5, Markers6):

        Os_AR = []
        Xs_AR = []
        Ys_AR = []
        Zs_AR = []
        Rs_AR = []

        for i in range(len(Markers1)):

            O_AR, X_AR, Y_AR, Z_AR, R_AR = Body_R.R_AR(Markers1[i], Markers2[i], Markers3[i],
                                                       Markers4[i], Markers5[i], Markers6[i])

            Os_AR.append(O_AR.tolist())
            Xs_AR.append(X_AR.tolist())
            Ys_AR.append(Y_AR.tolist())
            Zs_AR.append(Z_AR.tolist())
            Rs_AR.append(R_AR.tolist())

        Os_AR = np.array(Os_AR)
        Xs_AR = np.array(Xs_AR)
        Ys_AR = np.array(Ys_AR)
        Zs_AR = np.array(Zs_AR)
        Rs_AR = np.array(Rs_AR)

        return Os_AR, Xs_AR, Ys_AR, Zs_AR, Rs_AR
