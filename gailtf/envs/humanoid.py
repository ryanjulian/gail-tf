from gym.envs.mujoco import HumanoidEnv
from gym.envs.mujoco import mujoco_env
from gym import utils

import os
import numpy as np
from numpy.linalg import norm


def normalize(vector):
    return vector / norm(vector)


class HumanoidFeatureEnv(HumanoidEnv):
    def __init__(self):
        # The MuJoCo XML definition has been modified so that head, hands and feet are denoted as <body> elements
        # so that we can obtain their COMs via self.get_body_com(body_name).
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), 'humanoid_featurized.xml'), 5)
        self.model.opt.timestep = 0.005
        self.frame_skip = 6
        utils.EzPickle.__init__(self)

    def compute_features(self):
        features = np.array([
            normalize(self.get_body_com("head") - self.get_body_com("torso")),
            normalize(self.get_body_com("left_hand") - self.get_body_com("torso")),
            normalize(self.get_body_com("right_hand") - self.get_body_com("torso")),
            normalize(self.get_body_com("left_foot") - self.get_body_com("torso")),
            normalize(self.get_body_com("right_foot") - self.get_body_com("torso"))
        ])
        return features.flatten()

    def compute_positions(self):
        return {
            "root": self.get_body_com("torso"),
            "head": self.get_body_com("head"),
            "left_hand": self.get_body_com("left_hand"),
            "right_hand": self.get_body_com("right_hand"),
            "left_foot": self.get_body_com("left_foot"),
            "right_foot": self.get_body_com("right_foot")
        }
