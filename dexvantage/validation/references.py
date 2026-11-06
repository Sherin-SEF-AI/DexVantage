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
}


def cite(keys) -> list:
    """[(key, text)] for the keys known (unknown ones are left out)."""
    return [(k, REFERENCES[k]) for k in keys if k in REFERENCES]
