"""The literature, standards and software the checks and docs/overview.md cite, by key.

Ported from Vantage validation/references.py (the format; the entries are DexVantage's). Each phase adds the
references its checks cite.
"""

REFERENCES = {
    'moller1997': 'T. Möller and B. Trumbore, "Fast, minimum storage ray-triangle intersection", Journal of Graphics '
                  'Tools 2(1), 21–28, 1997.',
    'amanatides_woo1987': 'J. Amanatides and A. Woo, "A fast voxel traversal algorithm for ray tracing", '
                          'Eurographics \'87, 3–10, 1987.',
    'greene1986': 'N. Greene, "Environment mapping and other applications of world projections", IEEE Computer '
                  'Graphics and Applications 6(11), 21–29, 1986 (the cube map).',
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
    # optics (checks_optics.py)
    'opencv_calib3d': 'OpenCV, "Camera Calibration and 3D Reconstruction" (calib3d): the pinhole camera model with '
                      'Brown-Conrady radial, tangential and rational distortion (k1, k2, p1, p2, k3, k4, k5, k6) and '
                      'cv2.projectPoints, https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html.',
    'opencv_fisheye': 'OpenCV, "Fisheye camera model" (calib3d fisheye): θd = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸) '
                      'and cv2.fisheye.projectPoints, https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html.',
    'brown1966': 'D. C. Brown, "Decentering distortion of lenses", Photogrammetric Engineering 32(3), 444–462, 1966.',
    'kannala_brandt2006': 'J. Kannala and S. S. Brandt, "A generic camera model and calibration method for '
                          'conventional, wide-angle, and fish-eye lenses", IEEE Transactions on Pattern Analysis and '
                          'Machine Intelligence 28(8), 1335–1340, 2006.',
    'miyamoto1964': 'K. Miyamoto, "Fish eye lens", Journal of the Optical Society of America 54(8), 1060–1061, 1964.',
    'ray2002': 'S. F. Ray, Applied Photographic Optics, 3rd ed., Focal Press, 2002 (depth of field and hyperfocal '
               'distance of a thin lens).',
    'hecht2017': 'E. Hecht, Optics, 5th ed., Pearson, 2017 (the Gaussian thin-lens equation).',
    # depth validity (checks_depth.py)
    'gallup2008': 'D. Gallup, J.-M. Frahm, P. Mordohai and M. Pollefeys, "Variable baseline/resolution stereo", '
                  'IEEE CVPR 2008 (stereo depth error growing as z² / (b f)).',
    'intel_d400': 'RealSense, "RealSense Product Family D400 Series Datasheet", document 337029-017, revision 023, '
                  'March 2026 (Table 3-49 depth camera SKU properties, §4.4 depth FOV at distance, Table 4-12 '
                  'minimum-Z depth, Table 4-15 depth quality), https://www.realsenseai.com/wp-content/uploads/2026/03/'
                  'RealSense-D400-Series-Datasheet-Mar-2026.pdf, read 2026-11-06.',
    'intel_d457': 'RealSense, "RealSense Product Family D457 Datasheet", document 337029-011, revision 006, August '
                  '2025 (Table 3-19 depth camera D457 properties, Table 4-4 minimum-Z depth), https://realsenseai.com/'
                  'wp-content/uploads/2025/08/RealSense-D457-Datasheet-Aug_2025.pdf, read 2026-11-06.',
    # MuJoCo renders (checks_optics_mujoco.py, checks_occlusion_mujoco.py)
    'mujoco_camera': 'MuJoCo XML reference, body/camera (looks along its −z axis, +x right, +y up; fovy, focal, '
                     'focalpixel, principal, principalpixel, sensorsize, resolution) and visual/quality '
                     '(offsamples, numslices), doc/XMLreference.rst of https://github.com/google-deepmind/mujoco '
                     '(main branch), read 2026-11-06.',
    'mujoco_renderer': 'MuJoCo Python bindings, mujoco.Renderer (offscreen OpenGL rendering of RGB, depth as the '
                       'distance from the camera plane, and segmentation as (object id, object type)), with '
                       'MUJOCO_GL=egl, https://mujoco.readthedocs.io/en/stable/python.html#rendering.',
}


def cite(keys) -> list:
    """[(key, text)] for the keys known (unknown ones are left out)."""
    return [(k, REFERENCES[k]) for k in keys if k in REFERENCES]
