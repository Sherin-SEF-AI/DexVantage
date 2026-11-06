"""Robots: import (URDF, MJCF), the MuJoCo kinematics model and its shared cache.

    from dexvantage.core.robot import import_robot, get_model, RobotModel

import_robot reads a model file and every file it references into memory; RobotModel compiles them with MuJoCo (the
primary kinematics engine, spec §3) and gives forward kinematics, Jacobians and the link meshes posed at q.
"""
from .importer import RobotImportError, import_robot
from .model import RobotModel, RobotModelError, clear_cache, get_model

__all__ = ['RobotImportError', 'RobotModel', 'RobotModelError', 'clear_cache', 'get_model', 'import_robot']
