"""The literature, standards and software the checks and docs/overview.md cite, by key.

Ported from Vantage validation/references.py (the format; the entries are DexVantage's). Each phase adds the
references its checks cite.
"""

REFERENCES = {
    'rep103': 'ROS REP 103, Standard Units of Measure and Coordinate Conventions (T. Foote, M. Purvis, 2010), '
              'https://www.ros.org/reps/rep-0103.html.',
    'shoemake1985': 'K. Shoemake, "Animating rotation with quaternion curves", SIGGRAPH \'85, Computer Graphics 19(3), '
                    '245–254, 1985.',
    'shepperd1978': 'S. W. Shepperd, "Quaternion from rotation matrix", Journal of Guidance and Control 1(3), '
                    '223–224, 1978.',
    'scipy_rotation': 'SciPy, scipy.spatial.transform.Rotation (an independent implementation of rotation '
                      'conversions), https://docs.scipy.org/doc/scipy/reference/generated/'
                      'scipy.spatial.transform.Rotation.html.',
    'urdf': 'ROS URDF XML specification, the <origin rpy="…"> element: roll, pitch, yaw about fixed x, y, z axes, '
            'http://wiki.ros.org/urdf/XML/joint.',
    'mujoco_xml': 'MuJoCo XML reference: quat attributes are written (w, x, y, z), '
                  'https://mujoco.readthedocs.io/en/stable/XMLreference.html.',
    'hartley2004': 'R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed., Cambridge '
                   'University Press, 2004.',
    'json_schema': 'JSON Schema, draft 2020-12, https://json-schema.org/draft/2020-12.',
    'zip_appnote': 'PKWARE, APPNOTE.TXT, .ZIP File Format Specification (read here through Python\'s zipfile).',
    # kinematics (checks_kinematics.py)
    'lynch_park2017': 'K. M. Lynch and F. C. Park, Modern Robotics: Mechanics, Planning, and Control, Cambridge '
                      'University Press, 2017 (product of exponentials, chapter 4; space Jacobian, chapter 5).',
    'mujoco_todorov2012': 'E. Todorov, T. Erez and Y. Tassa, "MuJoCo: A physics engine for model-based control", '
                          'IEEE/RSJ IROS 2012, 5026–5033.',
    'dh1955': 'J. Denavit and R. S. Hartenberg, "A kinematic notation for lower-pair mechanisms based on matrices", '
              'Journal of Applied Mechanics 22(2), 215–221, 1955.',
    'spong2006': 'M. W. Spong, S. Hutchinson and M. Vidyasagar, Robot Modeling and Control, Wiley, 2006 (standard DH '
                 'link transform Rz(θ) Tz(d) Tx(a) Rx(α)).',
    'craig2005': 'J. J. Craig, Introduction to Robotics: Mechanics and Control, 3rd ed., Pearson Prentice Hall, 2005 '
                 '(modified DH link transform Rx(α) Tx(a) Rz(θ) Tz(d)).',
    'ur_dh': 'Universal Robots, "DH Parameters for calculations of kinematics and dynamics", UR5e/UR7e table, '
             'https://www.universal-robots.com/articles/ur/application-installation/'
             'dh-parameters-for-calculations-of-kinematics-and-dynamics/, read 2026-11-06.',
    'franka_dh': 'Franka Emika, "Robot and interface specifications", Denavit–Hartenberg parameters of the Panda '
                 '(Craig\'s convention), source/control_parameters.rst of https://github.com/frankarobotics/docs at '
                 'commit 476a2ed2c1f457b87ba2fbb9213ebcd13825382d (2021-11-11), read 2026-11-06; the page '
                 'https://frankaemika.github.io/docs/control_parameters.html returned HTTP 404 that day, and its '
                 'successor https://frankarobotics.github.io/docs/robot_specifications.html gives the same figures '
                 'for the Franka Research 3.',
    'pinocchio2019': 'J. Carpentier et al., "The Pinocchio C++ library: a fast and flexible implementation of rigid '
                     'body dynamics algorithms and their analytical derivatives", IEEE/SICE SII 2019, 614–619.',
    # inverse kinematics, manipulability and collision (checks_ik.py)
    'wampler1986': 'C. W. Wampler, "Manipulator inverse kinematic solutions based on vector formulations and damped '
                   'least-squares methods", IEEE Transactions on Systems, Man, and Cybernetics 16(1), 93–101, 1986.',
    'nakamura1986': 'Y. Nakamura and H. Hanafusa, "Inverse kinematic solutions with singularity robustness for robot '
                    'manipulator control", ASME Journal of Dynamic Systems, Measurement, and Control 108(3), 163–171, '
                    '1986.',
    'sugihara2011': 'T. Sugihara, "Solvability-unconcerned inverse kinematics by the Levenberg–Marquardt method", '
                    'IEEE Transactions on Robotics 27(5), 984–991, 2011.',
    'liegeois1977': 'A. Liégeois, "Automatic supervisory control of the configuration and behavior of multibody '
                    'mechanisms", IEEE Transactions on Systems, Man, and Cybernetics 7(12), 868–871, 1977.',
    'yoshikawa1985': 'T. Yoshikawa, "Manipulability of robotic mechanisms", The International Journal of Robotics '
                     'Research 4(2), 3–9, 1985.',
    'ericson2005': 'C. Ericson, Real-Time Collision Detection, Morgan Kaufmann, 2005 (§5.1.9, closest points of two '
                   'line segments).',
    'mujoco_geomdistance': 'MuJoCo API reference, mj_geomDistance (the signed distance between two geoms, negative '
                           'when they overlap), https://mujoco.readthedocs.io/en/stable/APIreference/'
                           'APIfunctions.html#mj-geomdistance.',
}


def cite(keys) -> list:
    """[(key, text)] for the keys known (unknown ones are left out)."""
    return [(k, REFERENCES[k]) for k in keys if k in REFERENCES]
