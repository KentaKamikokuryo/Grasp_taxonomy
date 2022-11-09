import numpy as np


class R_utilities():

    @staticmethod
    def compute_fixed_position(position_target: np.ndarray, R_body: np.ndarray):

        position_fixed = np.dot(position_target, R_body)

        return position_fixed

    @staticmethod
    def compute_fixed_positions(positions_target: np.ndarray, Rs_body: np.ndarray):

        positions_fixed = []

        for i in range(len(positions_target)):

            position_fixed = R_utilities.compute_fixed_position(position_target=positions_target[i], R_body=Rs_body[i])

            positions_fixed.append(position_fixed.tolist())

        positions_fixed = np.array(positions_fixed)

        return positions_fixed
