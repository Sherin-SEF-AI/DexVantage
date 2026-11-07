"""The literature, standards and software the checks and docs/overview.md cite, by key.

Ported from Vantage validation/references.py (the format; the entries are DexVantage's). Each phase adds the
references its checks cite.
"""

REFERENCES = {
    'arvo1990': 'J. Arvo, "Transforming axis-aligned bounding boxes", in A. S. Glassner (ed.), Graphics Gems, '
                'Academic Press, 548–550, 1990.',
    'mcp_2025_11_25': 'Model Context Protocol specification, version 2025-11-25: basic/lifecycle, basic/transports '
                      '(stdio), server/tools, https://modelcontextprotocol.io/specification/2025-11-25 (read '
                      '2026-11-07).',
    'jsonrpc2': 'JSON-RPC 2.0 Specification (JSON-RPC Working Group, 2010), https://www.jsonrpc.org/specification.',
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
    'hecht2017': 'E. Hecht, Optics, 5th ed., Pearson, 2017 (the Gaussian thin-lens equation; the Fresnel equations, '
                 '§4.6; Malus\'s law, §8.2).',
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
    # policy inputs (checks_policies.py)
    'openpi': 'Physical Intelligence, openpi, src/openpi/shared/image_tools.py resize_with_pad and '
              'src/openpi/models/model.py, https://github.com/Physical-Intelligence/openpi (commit 215abfb), '
              'read 2026-11-06.',
    'lerobot': 'Hugging Face, LeRobot, src/lerobot/policies/common/vla_utils.py resize_with_pad (left and top '
               'padding), https://github.com/huggingface/lerobot (commit d490e47), read 2026-11-06.',
    'torchvision_centercrop': 'torchvision.transforms.functional.center_crop (crop_top = int(round((image_height − '
                              'crop_height) / 2.0)), likewise the left), torchvision/transforms/functional.py of '
                              'https://github.com/pytorch/vision (main branch), read 2026-11-06.',
    'isaac_gr00t': 'NVIDIA, Isaac-GR00T, branch n1-release, gr00t/data/transform/video.py VideoCrop and '
                   'gr00t/experiment/data_config.py, https://github.com/NVIDIA/Isaac-GR00T (commit 755876a), '
                   'read 2026-11-06.',
    'opencv_resize': 'OpenCV, cv2.resize with INTER_LINEAR (bilinear, pixel centres aligned), '
                     'https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html.',
    # VLA tokens and Plücker ray maps (checks_vla.py)
    'plucker1865': 'J. Plücker, "On a new geometry of space", Philosophical Transactions of the Royal Society of '
                   'London 155, 725–791, 1865 (line coordinates: direction d and moment m = o × d, m · d = 0).',
    'safetensors': 'Hugging Face, safetensors, README "Format" (8-byte little-endian header length, UTF-8 JSON '
                   'header with dtype, shape and data_offsets, __metadata__, no holes, little-endian row-major) and '
                   'safetensors/src/tensor.rs (dtype names, header padded with spaces to 8 bytes), '
                   'https://github.com/huggingface/safetensors, read 2026-11-06.',
    # cables across joints (checks_cables.py)
    'docarmo1976': 'M. P. do Carmo, Differential Geometry of Curves and Surfaces, Prentice-Hall, 1976 (§1-5: the '
                   'helix (a cos s/c, a sin s/c, b s/c), c² = a² + b², has curvature a / c²).',
    'scipy_brentq': 'SciPy, scipy.optimize.brentq (Brent\'s root finder, an independent solver), '
                    'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html.',
    # bandwidth, hosts and power (checks_bandwidth.py)
    'usb32_spec': 'USB 3.0 Promoter Group, Universal Serial Bus 3.2 Specification, Revision 1.1, June 2022 (Table 3-1 '
                  'bus power, §4.4.11 efficiency, §10.14 hub configuration and §11.4 power distribution), '
                  'https://www.usb.org/sites/default/files/usb_32_202206_0.zip, read 2026-11-06.',
    'usb20_spec': 'Universal Serial Bus Specification, Revision 2.0, April 27, 2000 (Table 5-10 high-speed bulk '
                  'transaction limits, §7.2.1 classes of devices), '
                  'https://www.usb.org/sites/default/files/usb_20_20250603.zip, read 2026-11-06.',
    'naneyec': 'ams OSRAM, NanEyeC miniature camera module datasheet (frame rate and serial data rate over SEIM and '
               'LVDS), https://look.ams-osram.com/m/19863b4335e67f1b/original/NanEyeC-Miniature-Camera-Module.pdf, '
               'as recorded in the library entry ams-osram-naneyec (read 2026-11-06).',
    # brackets (checks_brackets.py)
    'treece1999': 'G. M. Treece, R. W. Prager and A. H. Gee, "Regularised marching tetrahedra: improved iso-surface '
                  'extraction", Computers & Graphics 23(4), 583–598, 1999.',
    'euler_poincare': 'The Euler–Poincaré formula for a closed orientable surface of genus g: V − E + F = 2 − 2g per '
                      'connected shell (e.g. M. de Berg et al., Computational Geometry, 3rd ed., Springer, 2008, '
                      '§2.2).',
    'ur5e_flange': 'Universal Robots, UR5e datasheet, drawing "Tool flange for UR5e" (EN ISO-9409-1-50-4-M6), '
                   'https://www.universal-robots.com/manuals/latest/en/datasheets/ur5e/, read 2026-11-06.',
    'iso9409_1': 'ISO 9409-1:2004, Manipulating industrial robots — Mechanical interfaces — Part 1: Plates; public '
                 'preview (clauses 4 and 5, Figure 1), https://cdn.standards.iteh.ai/samples/36578/'
                 '348a837664f444bc83e2902c7a5acf4c/ISO-9409-1-2004.pdf, read 2026-11-06.',
    # hand-eye calibration (checks_handeye.py, core/calib)
    'tsai_lenz1989': 'R. Y. Tsai and R. K. Lenz, "A new technique for fully autonomous and efficient 3D robotics '
                     'hand/eye calibration", IEEE Transactions on Robotics and Automation 5(3), 345–358, 1989.',
    'park_martin1994': 'F. C. Park and B. J. Martin, "Robot sensor calibration: solving AX = XB on the Euclidean '
                       'group", IEEE Transactions on Robotics and Automation 10(5), 717–721, 1994.',
    'horaud_dornaika1995': 'R. Horaud and F. Dornaika, "Hand-eye calibration", The International Journal of Robotics '
                           'Research 14(3), 195–210, 1995.',
    'andreff1999': 'N. Andreff, R. Horaud and B. Espiau, "On-line hand-eye calibration", Second International '
                   'Conference on 3-D Digital Imaging and Modeling (3DIM), 430–436, 1999.',
    'daniilidis1999': 'K. Daniilidis, "Hand-eye calibration using dual quaternions", The International Journal of '
                      'Robotics Research 18(3), 286–298, 1999.',
    'shah2013': 'M. Shah, "Solving the robot-world/hand-eye calibration problem using the Kronecker product", Journal '
                'of Mechanisms and Robotics 5(3), 031007, 2013.',
    'li2010': 'A. Li, L. Wang and D. Wu, "Simultaneous robot-world and hand-eye calibration using dual-quaternions and '
              'Kronecker product", International Journal of the Physical Sciences 5(10), 1530–1536, 2010.',
    'opencv_handeye': 'OpenCV calib3d, calibrateHandEye and calibrateRobotWorldHandEye (conventions: gripper-to-base '
                      'and target-to-camera in, camera-to-gripper out; world-to-camera and base-to-gripper in, '
                      'base-to-world and gripper-to-camera out), https://docs.opencv.org/4.x/d9/d0c/'
                      'group__calib3d.html.',
    'triggs2000': 'B. Triggs, P. McLauchlan, R. Hartley and A. Fitzgibbon, "Bundle adjustment — a modern synthesis", '
                  'Vision Algorithms: Theory and Practice, LNCS 1883, 298–372, Springer, 2000.',
    'garrido2014': 'S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas and M. J. Marín-Jiménez, "Automatic '
                   'generation and detection of highly reliable fiducial markers under occlusion", Pattern '
                   'Recognition 47(6), 2280–2292, 2014.',
    'opencv_charuco': 'OpenCV objdetect aruco module: CharucoBoard, CharucoDetector, ArucoDetector and the predefined '
                      'dictionaries (DICT_APRILTAG_36h11 among them), https://docs.opencv.org/4.x/df/d4a/'
                      'tutorial_charuco_detection.html.',
    'wang_olson2016': 'J. Wang and E. Olson, "AprilTag 2: Efficient and robust fiducial detection", IEEE/RSJ IROS '
                      '2016, 4193–4198.',
    'kalibr': 'ethz-asl/kalibr: the "Calibration targets" wiki page (AprilGrid tagSize "edge to edge", tagSpacing '
              'the ratio of the gap to tagSize, 0.3 in the example), aslam_cv/aslam_cameras_april/src/'
              'GridCalibrationTargetAprilgrid.cpp and its header (point order, blackTagBorder 2, minTagsForValidObs '
              '4) and kalibr_create_target_pdf (tag ids row by row from the bottom left, corner squares), '
              'https://github.com/ethz-asl/kalibr, read 2026-11-06.',
    # grasp self-occlusion (checks_grasp.py)
    'efron1979': 'B. Efron, "Bootstrap methods: another look at the jackknife", The Annals of Statistics 7(1), 1–26, '
                 '1979 (the percentile bootstrap interval).',
    'wilson1927': 'E. B. Wilson, "Probable inference, the law of succession, and statistical inference", Journal of '
                  'the American Statistical Association 22(158), 209–212, 1927 (the score interval of a proportion).',
    # LeRobot datasets (checks_formats.py formats.lerobot_roundtrip)
    'lerobot_dataset_format': 'Hugging Face, LeRobot dataset format: v3.0 from src/lerobot/datasets/{utils, '
                              'dataset_metadata, dataset_writer, io_utils, compute_stats, video_utils}.py, '
                              'src/lerobot/scripts/convert_dataset_v21_to_v30.py and docs/source/'
                              'lerobot-dataset-v3.mdx at commit d490e472a5d1a207cfe0255caaaf74b6d66e4656; v2.1 from '
                              'the same files at tag v0.3.3 (commit b883328e6c95681ca90a18b102e4ae5e1f91e2bf), '
                              'https://github.com/huggingface/lerobot, read 2026-11-07.',
    'apache_parquet': 'Apache Parquet file format, https://parquet.apache.org/docs/file-format/ (read here through '
                      'pyarrow.parquet, https://arrow.apache.org/docs/python/parquet.html).',
    'ffmpeg_cli': 'FFmpeg command-line tools ffmpeg and ffprobe (rawvideo rgb24 output, frame pts_time), '
                  'https://ffmpeg.org/ffmpeg.html and https://ffmpeg.org/ffprobe.html.',
    # ROS 2 bags and RLDS (checks_formats.py, formats_bags.py; core/rosbag.py, core/datasets/bag.py and rlds.py)
    'mcap_spec': 'Foxglove, MCAP file format specification (records, chunks, message indexes, summary, CRC-32 fields), '
                 'https://mcap.dev/spec, read 2026-11-07.',
    'mcap_python': 'Foxglove, mcap Python package (mcap.reader.make_reader with validate_crcs), MIT licence, '
                   'https://github.com/foxglove/mcap/tree/main/python/mcap, version 1.6.0 used 2026-11-07.',
    'rosbag2_mcap': 'ROS 2 rosbag2 storage plugins rosbag2_storage_mcap (profile ros2, ros2msg schemas, cdr messages) '
                    'and rosbag2_storage_sqlite3 (topics and messages tables), https://github.com/ros2/rosbag2.',
    'omg_xtypes_cdr': 'Object Management Group, DDS-XTypes 1.3, section 7.4 (XCDR1 "plain CDR" serialisation: '
                      'encapsulation header, alignment of primitives to their size), '
                      'https://www.omg.org/spec/DDS-XTypes/1.3.',
    'rfc3720': 'J. Satran et al., RFC 3720, Internet Small Computer Systems Interface (iSCSI), April 2004, appendix '
               'B.4 CRC examples (CRC32C test vectors), https://www.rfc-editor.org/rfc/rfc3720.txt, read 2026-11-07.',
    'crc_catalogue': 'G. Cook, Catalogue of parametrised CRC algorithms, CRC-32/ISCSI (CRC-32C, Castagnoli): poly '
                     '0x1edc6f41, init and xorout 0xffffffff, reflected, check 0xe3069283, '
                     'https://reveng.sourceforge.io/crc-catalogue/17plus.htm, read 2026-11-07.',
    'tfrecord_format': 'TensorFlow TFRecord format: uint64 length, uint32 masked CRC-32C of the length, data, uint32 '
                       'masked CRC-32C of the data; mask = rotate right 15 + 0xa282ead8 (xla/tsl/lib/io/'
                       'record_writer.h and xla/tsl/lib/hash/crc32c.h, https://github.com/openxla/xla; '
                       'https://www.tensorflow.org/tutorials/load_data/tfrecord), read 2026-11-07.',
    'tf_example_proto': 'TensorFlow tensorflow/core/example/example.proto and feature.proto (Example, '
                        'SequenceExample, Features, Feature, BytesList / FloatList / Int64List), '
                        'https://github.com/tensorflow/tensorflow, read 2026-11-07.',
    'protobuf_encoding': 'Protocol Buffers encoding guide (varints, wire types, packed repeated fields, maps), '
                         'https://protobuf.dev/programming-guides/encoding/, read 2026-11-07.',
    'tfds_serializer': 'TensorFlow Datasets tensorflow_datasets/core/example_serializer.py and features/ '
                       '(flattened feature names, stacked sequences, Tensor encodings none / bytes / zlib, float32 '
                       'images bitcast to 4-channel PNG), https://github.com/tensorflow/datasets, read 2026-11-07.',
    'rlds2021': 'S. Ramos et al., "RLDS: an ecosystem to generate, share and use datasets in reinforcement learning", '
                'arXiv:2111.02767, 2021; dataset format in https://github.com/google-research/rlds (README), read '
                '2026-11-07.',
    'lerobot_so100': 'Hugging Face, LeRobot documentation, "SO-100" (the leader arm is built from the follower\'s '
                     'parts through the wrist and both are calibrated to the same joint positions), '
                     'https://huggingface.co/docs/lerobot/so100, read 2026-11-07.',
    'gello2023': 'P. Wu, Y. Shentu, Z. Yi, X. Lin and P. Abbeel, "GELLO: A General, Low-Cost, and Intuitive '
                 'Teleoperation Framework for Robot Manipulators", arXiv:2309.13037, 2023.',
    # action-chunk latency (checks_chunk.py)
    'zhao2023_act': 'T. Z. Zhao, V. Kumar, S. Levine and C. Finn, "Learning fine-grained bimanual manipulation with '
                    'low-cost hardware" (ACT, temporal ensembling w_i = exp(-m i)), RSS 2023, '
                    'https://arxiv.org/abs/2304.13705.',
    'act_code': 'ACT official code, imitate_episodes.py, temporal_agg (all_time_actions rows by issue step, '
                'exp_weights = np.exp(-k * np.arange(n)), k = 0.01: row 0, the oldest chunk, weighs most), '
                'https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py, '
                'read 2026-11-07.',
    'lerobot_act_ensembler': 'Hugging Face, LeRobot, src/lerobot/policies/act/modeling_act.py ACTTemporalEnsembler '
                             '("w_0 is the oldest action"), https://github.com/huggingface/lerobot/blob/'
                             'd490e472a5d1a207cfe0255caaaf74b6d66e4656/src/lerobot/policies/act/modeling_act.py, '
                             'read 2026-11-07.',
    # lighting, glare and IR (checks_lighting.py, core/materials.py, core/lights.py, core/studies/lighting.py)
    'walter2007': 'B. Walter, S. R. Marschner, H. Li and K. E. Torrance, "Microfacet models for refraction through '
                  'rough surfaces", Eurographics Symposium on Rendering 2007, 195–206 (the GGX distribution, eq. 33; '
                  'Smith G1, eq. 34; the microfacet BRDF, eq. 20; sampling, eqs. 35–36).',
    'trowbridge1975': 'T. S. Trowbridge and K. P. Reitz, "Average irregularity representation of a rough surface for '
                      'ray reflection", Journal of the Optical Society of America 65(5), 531–536, 1975.',
    'smith1967': 'B. Smith, "Geometrical shadowing of a random rough surface", IEEE Transactions on Antennas and '
                 'Propagation 15(5), 668–671, 1967.',
    'schlick1994': 'C. Schlick, "An inexpensive BRDF model for physically-based rendering", Computer Graphics Forum '
                   '13(3), 233–246, 1994 (the Fresnel approximation, eq. 15).',
    'ashikhmin2000': 'M. Ashikhmin and P. Shirley, "An anisotropic Phong BRDF model", Journal of Graphics Tools 5(2), '
                     '25–32, 2000 (the energy-coupled diffuse term, eq. 5).',
    'heitz2014': 'E. Heitz, "Understanding the masking-shadowing function in microfacet-based BRDFs", Journal of '
                 'Computer Graphics Techniques 3(2), 48–107, 2014 (the weak white furnace test).',
    'veach1997': 'E. Veach, Robust Monte Carlo Methods for Light Transport Simulation, PhD thesis, Stanford '
                 'University, 1997 (multiple importance sampling, chapter 9).',
    'matusik2003': 'W. Matusik, H. Pfister, M. Brand and L. McMillan, "A data-driven reflectance model", ACM '
                   'Transactions on Graphics 22(3), 759–769, 2003 (the MERL BRDF database).',
    'lowe2012': 'J. Löw, J. Kronander, A. Ynnerman and J. Unger, "BRDF models for accurate and efficient rendering of '
                'glossy surfaces", ACM Transactions on Graphics 31(1), 9, 2012 (microfacet fits to MERL).',
    'sultanova2009': 'N. Sultanova, S. Kasarova and I. Nikolov, "Dispersion properties of optical polymers", Acta '
                     'Physica Polonica A 116, 585–587, 2009 (PMMA; via the refractiveindex.info database, CC0, read '
                     '2026-11-07).',
    'schott_bk7': 'SCHOTT optical glass catalogue 2017-01-20b, N-BK7 Sellmeier coefficients (via the '
                  'refractiveindex.info database, CC0, read 2026-11-07).',
    'mcpeak2015': 'K. M. McPeak et al., "Plasmonic films can easily be better: rules and recipes", ACS Photonics 2, '
                  '326–333, 2015 (aluminium n, k; via the refractiveindex.info database, CC0, read 2026-11-07).',
    'mcluney2014': 'W. R. McCluney, Introduction to Radiometry and Photometry, 2nd ed., Artech House, 2014 (the '
                   'inverse-square law, Lambertian sources).',
    'howell_catalog': 'J. R. Howell, A Catalog of Radiation Heat Transfer Configuration Factors, configuration B-3 '
                      '(plane element parallel to a rectangle, its normal through a corner), '
                      'http://www.thermalradiation.net/indexCat.html.',
    'iso12232': 'ISO 12232, Photography — Digital still cameras — Determination of exposure index, ISO speed '
                'ratings, standard output sensitivity and recommended exposure index (saturation-based speed '
                '78 lx·s / H_sat: 18 % grey at 18 %/√2 of saturation), as described in Wikipedia, "Film speed", read '
                '2026-11-07.',
    'intel_d400_projector': 'RealSense Product Family D400 Series Datasheet, document 337029-017, March 2026, Tables '
                            '3-16 and 3-17 (projector optical power 360 mW average, 850 nm), read 2026-11-07.',
    'williams1978': 'L. Williams, "Casting curved shadows on curved surfaces", SIGGRAPH \'78, Computer Graphics 12(3), '
                    '270–274, 1978 (shadow maps and their depth bias).',
    'scipy_quad': 'SciPy, scipy.integrate.quad (adaptive quadrature, QUADPACK), '
                  'https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html.',
    # motion blur and time sync (checks_blur.py, checks_sync.py)
    'liang2008': 'C.-K. Liang, L.-W. Chang and H. H. Chen, "Analysis and compensation of rolling shutter effect", '
                 'IEEE Transactions on Image Processing 17(8), 1323–1330, 2008.',
    'ms_usb_isoch': 'Microsoft, "How to transfer data to USB isochronous endpoints", Windows driver documentation, '
                    '2024-01-17: "For high speed and SuperSpeed, the bus interval is a microframe. The length of a '
                    'microframe is 125 microseconds", https://learn.microsoft.com/windows-hardware/drivers/usbcon/'
                    'transfer-data-to-isochronous-endpoints (read 2026-11-07).',
    # MuJoCo ground truth (checks_groundtruth.py)
    'mujoco_camera_317': 'MuJoCo XML reference at tag 3.17.0 (the installed version), body/camera: fovy, resolution, '
                         'sensorsize ("When specified, all intrinsic attributes become active and fovy is ignored"), '
                         'focal / focalpixel, principal / principalpixel ("Offset of the principal point ... from '
                         'the image center"), and visual/map znear, zfar; doc/XMLreference.rst of '
                         'https://github.com/google-deepmind/mujoco/blob/3.17.0/doc/XMLreference.rst, read '
                         '2026-11-07.',
    'opencv_remap': 'OpenCV imgproc, cv2.remap (dst(x, y) = src(map_x(x, y), map_y(x, y)), nearest and bilinear '
                    'interpolation), https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html.',
    # the dataset auditor (checks_audit.py, core/audit)
    'knapp_carter1976': 'C. H. Knapp and G. C. Carter, "The generalized correlation method for estimation of time '
                        'delay", IEEE Transactions on Acoustics, Speech, and Signal Processing 24(4), 320–327, 1976.',
    'spearman1904': 'C. Spearman, "The proof and measurement of association between two things", The American '
                    'Journal of Psychology 15(1), 72–101, 1904 (the rank correlation).',
    'kunsch1989': 'H. R. Künsch, "The jackknife and the bootstrap for general stationary observations", The Annals of '
                  'Statistics 17(3), 1217–1241, 1989 (the moving-block bootstrap).',
    'kass_raftery1995': 'R. E. Kass and A. E. Raftery, "Bayes factors", Journal of the American Statistical '
                        'Association 90(430), 773–795, 1995 (BIC differences over 6: strong evidence).',
    'lucas_kanade1981': 'B. D. Lucas and T. Kanade, "An iterative image registration technique with an application '
                        'to stereo vision", IJCAI 1981, 674–679.',
    'baker_matthews2004': 'S. Baker and I. Matthews, "Lucas-Kanade 20 years on: a unifying framework", International '
                          'Journal of Computer Vision 56(3), 221–255, 2004.',
    'pech_pacheco2000': 'J. L. Pech-Pacheco, G. Cristóbal, J. Chamorro-Martínez and J. Fernández-Valdivia, '
                        '"Diatom autofocusing in brightfield microscopy: a comparative study", ICPR 2000, vol. 3, '
                        '314–317 (the variance of the Laplacian as a focus measure).',
    'huber1964': 'P. J. Huber, "Robust estimation of a location parameter", The Annals of Mathematical Statistics '
                 '35(1), 73–101, 1964.',
    'fiftyone_format': 'Voxel51, FiftyOne, fiftyone/utils/data/importers.py (FiftyOneDatasetImporter, '
                       'LegacyFiftyOneDatasetImporter) and fiftyone/core/sample.py (Sample.from_dict), '
                       'https://github.com/voxel51/fiftyone (branch develop), and the user guide "Exporting FiftyOne '
                       'Datasets", https://docs.voxel51.com/user_guide/export_datasets.html, read 2026-11-07.',
    # targetless link-silhouette calibration (checks_render_calib.py, core/calib/render_calib.py)
    'barrow1977': 'H. G. Barrow, J. M. Tenenbaum, R. C. Bolles and H. C. Wolf, "Parametric correspondence and chamfer '
                  'matching: two new techniques for image matching", Proc. 5th IJCAI, 659–663, 1977.',
    'borgefors1988': 'G. Borgefors, "Hierarchical chamfer matching: a parametric edge matching algorithm", IEEE '
                     'Transactions on Pattern Analysis and Machine Intelligence 10(6), 849–865, 1988.',
    'canny1986': 'J. Canny, "A computational approach to edge detection", IEEE Transactions on Pattern Analysis and '
                 'Machine Intelligence 8(6), 679–698, 1986.',
    'liu2010_fdcm': 'M.-Y. Liu, O. Tuzel, A. Veeraraghavan and R. Chellappa, "Fast directional chamfer matching", '
                    'IEEE CVPR 2010, 1696–1703 (edges matched within orientation channels).',
    'opencv_distance_transform': 'OpenCV imgproc, cv2.distanceTransform and cv2.distanceTransformWithLabels '
                                 '(DIST_L2 with a 5 × 5 mask, DIST_LABEL_PIXEL), '
                                 'https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html.',
    'liang_zeger1986': 'K.-Y. Liang and S. L. Zeger, "Longitudinal data analysis using generalized linear models", '
                       'Biometrika 73(1), 13–22, 1986 (the cluster-robust "sandwich" covariance).',
    # designed experiments (checks_experiments.py, core/experiments.py)
    'sobol1993': 'I. M. Sobol\', "Sensitivity estimates for nonlinear mathematical models", Mathematical Modelling '
                 'and Computational Experiments 1(4), 407–414, 1993.',
    'saltelli2002': 'A. Saltelli, "Making best use of model evaluations to compute sensitivity indices", Computer '
                    'Physics Communications 145(2), 280–297, 2002.',
    'saltelli2010': 'A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto and S. Tarantola, "Variance based '
                    'sensitivity analysis of model output. Design and estimator for the total sensitivity index", '
                    'Computer Physics Communications 181(2), 259–270, 2010 (Table 2: the estimators).',
    'jansen1999': 'M. J. W. Jansen, "Analysis of variance designs for model output", Computer Physics Communications '
                  '117(1–2), 35–43, 1999.',
    'ishigami1990': 'T. Ishigami and T. Homma, "An importance quantification technique in uncertainty analysis for '
                    'computer models", Proc. First International Symposium on Uncertainty Modeling and Analysis '
                    '(ISUMA \'90), 398–403, IEEE, 1990.',
    'mckay1979': 'M. D. McKay, R. J. Beckman and W. J. Conover, "A comparison of three methods for selecting values '
                 'of input variables in the analysis of output from a computer code", Technometrics 21(2), 239–245, '
                 '1979.',
    'hickernell1998': 'F. J. Hickernell, "A generalized discrepancy and quadrature error bound", Mathematics of '
                      'Computation 67(221), 299–322, 1998 (the centred L2 discrepancy).',
    'box_hunter2005': 'G. E. P. Box, J. S. Hunter and W. G. Hunter, Statistics for Experimenters: Design, '
                      'Innovation, and Discovery, 2nd ed., Wiley, 2005 (two-level factorials and their fractions, '
                      'resolution).',
    'efron1993': 'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall, 1993 (percentile '
                 'intervals).',
    'scipy_qmc': 'SciPy, scipy.stats.qmc (Sobol with Owen scrambling, discrepancy), '
                 'https://docs.scipy.org/doc/scipy/reference/stats.qmc.html.',
    # Monte Carlo tolerances and the perturbation export (checks_tolerance.py, core/tolerance.py)
    'jcgm100_2008': 'JCGM 100:2008, Evaluation of measurement data — Guide to the expression of uncertainty in '
                    'measurement (GUM), clause 5.1.2 and 5.2.2, eq. 10 and 13 (the law of propagation of '
                    'uncertainty), https://www.bipm.org/en/committees/jc/jcgm/publications.',
    'jcgm101_2008': 'JCGM 101:2008, Evaluation of measurement data — Supplement 1 to the GUM: propagation of '
                    'distributions using a Monte Carlo method, https://www.bipm.org/en/committees/jc/jcgm/'
                    'publications.',
    'kolmogorov1933': 'A. N. Kolmogorov, "Sulla determinazione empirica di una legge di distribuzione", Giornale '
                      'dell\'Istituto Italiano degli Attuari 4, 83–91, 1933.',
    'smirnov1948': 'N. Smirnov, "Table for estimating the goodness of fit of empirical distributions", The Annals of '
                   'Mathematical Statistics 19(2), 279–281, 1948.',
    'scipy_stats': 'SciPy, scipy.stats (kstest, uniform, truncnorm, norm, spearmanr: independent implementations), '
                   'https://docs.scipy.org/doc/scipy/reference/stats.html.',
    'isaaclab_camera_cfg': 'Isaac Lab, source/isaaclab/isaaclab/sensors/camera/camera_cfg.py, CameraCfg.OffsetCfg '
                           '(pos, rot, convention "ros"; rot (w, x, y, z) at tag v2.3.0, (x, y, z, w) at tag v3.0.0 '
                           'and main commit d4be263), https://github.com/isaac-sim/IsaacLab, read 2026-11-07.',
    'kerbl2023_3dgs': 'B. Kerbl, G. Kopanas, T. Leimkühler and G. Drettakis, "3D Gaussian Splatting for real-time '
                      'radiance field rendering", ACM Transactions on Graphics 42(4), 2023; reference code '
                      'utils/camera_utils.py camera_to_JSON, https://github.com/graphdeco-inria/gaussian-splatting '
                      '(main, commit 54c035f), read 2026-11-07.',
    # the canonical view warp and the Rerun bridge (checks_warp.py, checks_rerun.py)
    'irani_anandan1996': 'M. Irani and P. Anandan, "Parallax geometry of pairs of points for 3D scene analysis", ECCV '
                         '1996, LNCS 1064, 17–30, doi:10.1007/BFb0015520 (the plane-plus-parallax decomposition; '
                         'bibliographic details read at dblp.org, 2026-11-07).',
    'opencv_warp_perspective': 'OpenCV imgproc, cv2.warpPerspective (dst(x, y) = src(M⁻¹ (x, y)) unless '
                               'WARP_INVERSE_MAP is set; sample positions in 1/INTER_TAB_SIZE = 1/32 px), the '
                               'docstring of the installed OpenCV 5.0.0, read 2026-11-07; '
                               'https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html returned HTTP 403 '
                               'that day.',
    'opencv_filestorage': 'OpenCV core, cv::FileStorage (XML, YAML and JSON persistence of matrices), '
                          'https://docs.opencv.org/4.x/da/d56/classcv_1_1FileStorage.html (HTTP 403 on 2026-11-07; '
                          'the installed OpenCV 5.0.0 bindings were used).',
    'rerun_sdk': 'Rerun, rerun-sdk 0.42.0 for Python (RecordingStream, the archetypes Transform3D, Pinhole, Mesh3D, '
                 'Boxes3D, TextLog, Scalars, and the reader rerun.chunk.RrdReader): the installed wheel\'s source and '
                 'https://rerun.io/docs/reference/types/archetypes/pinhole (version 0.42.0), read 2026-11-07.',
    # the optimiser and the design agent (checks_optimise.py)
    'deb2002_nsga2': 'K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic '
                     'algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation 6(2), 182–197, 2002 (the '
                     'fast non-dominated sort, crowding distance, constrained domination and the CONSTR problem).',
    'deb_agrawal1995_sbx': 'K. Deb and R. B. Agrawal, "Simulated binary crossover for continuous search space", '
                           'Complex Systems 9(2), 115–148, 1995.',
    'deb_goyal1996_mutation': 'K. Deb and M. Goyal, "A combined genetic adaptive search (GeneAS) for engineering '
                              'design", Computer Science and Informatics 26(4), 30–45, 1996 (polynomial mutation).',
    'zitzler2000_zdt': 'E. Zitzler, K. Deb and L. Thiele, "Comparison of multiobjective evolutionary algorithms: '
                       'empirical results", Evolutionary Computation 8(2), 173–195, 2000 (ZDT1 and ZDT2 and their '
                       'analytic fronts, g = 1).',
    'deb2001_moo': 'K. Deb, Multi-Objective Optimization Using Evolutionary Algorithms, Wiley, 2001 (constrained test '
                   'problems).',
    'coello2004_igd': 'C. A. Coello Coello and M. Reyes Sierra, "A study of the parallelization of a coevolutionary '
                      'multi-objective evolutionary algorithm", MICAI 2004, LNAI 2972, 688–697 (inverted '
                      'generational distance).',
    # the assistant dock (checks_copilot.py)
    'fips180_4': 'NIST FIPS PUB 180-4, Secure Hash Standard (SHS), August 2015 (SHA-256), as implemented by Python\'s '
                 'hashlib, https://csrc.nist.gov/pubs/fips/180-4/upd1/final (read 2026-11-07).',
    # exporters part A (checks_formats.py, core/exporters: mjcf, ros2, calibration, masks)
    'mujoco_urdf': 'MuJoCo XML reference and modeling guide, URDF extensions: the <mujoco><compiler …/></mujoco> '
                   'element of a URDF (fusestatic, discardvisual, boundmass), read through mujoco.MjModel at 3.17.0, '
                   'https://mujoco.readthedocs.io/en/stable/modeling.html#urdf-extensions.',
    'urdfdom': 'ros/urdfdom at commit bfcf29f (urdf_parser/src/joint.cpp: <limit> needs effort and velocity) and '
               'ros/urdfdom_headers (urdf_model/pose.h, Rotation::setFromRPY: fixed-axis roll, pitch, yaw), '
               'https://github.com/ros/urdfdom, read 2026-11-07.',
    'tf2_static_publisher': 'ros2/geometry2 at commit 183399c, tf2_ros/src/static_transform_broadcaster_program.cpp '
                            '(--x --y --z --qx --qy --qz --qw --frame-id --child-frame-id), '
                            'https://github.com/ros2/geometry2, read 2026-11-07.',
    'kalibr_yaml': 'ethz-asl/kalibr wiki, "YAML formats" (camchain.yaml: camera_model, intrinsics [fu fv pu pv], '
                   'distortion_model, distortion_coeffs, resolution, rostopic, T_cn_cnm1) and "Supported models"; '
                   'kalibr_common/ConfigReader.py and aslam_cv RadialTangentialDistortion / EquidistantDistortion '
                   'at commit 1f60227, https://github.com/ethz-asl/kalibr, read 2026-11-07.',
    'pyyaml': 'PyYAML (yaml.safe_load), https://pyyaml.org/wiki/PyYAMLDocumentation.',
    'python_ast': 'Python standard library, ast (parse, literal_eval) and xml.etree.ElementTree, '
                  'https://docs.python.org/3/library/ast.html.',
    'gmsh': 'C. Geuzaine and J.-F. Remacle, "Gmsh: a 3-D finite element mesh generator with built-in pre- and '
            'post-processing facilities", International Journal for Numerical Methods in Engineering 79(11), '
            '1309–1331, 2009; gmsh 4.15.2 Python API (occ.addPlaneSurface, addSurfaceLoop with sewing, '
            'importShapes, getMass), https://gmsh.info.',
    'step_ap': 'ISO 10303-21, Industrial automation systems and integration: product data representation and exchange, '
               'clear text encoding of the exchange structure (STEP), as written and read by OpenCASCADE through '
               'gmsh.',
    'posix_exit': 'The Open Group Base Specifications Issue 7, 2018 edition (IEEE Std 1003.1-2017), exit(): "The value '
                  'of status may be 0, EXIT_SUCCESS, EXIT_FAILURE, or any other value", '
                  'https://pubs.opengroup.org/onlinepubs/9699919799/functions/exit.html (read 2026-11-07); the meaning '
                  'of 1 and 2 is DexVantage\'s (spec §11).',
    'metropolis_ulam1949': 'N. Metropolis and S. Ulam, "The Monte Carlo method", Journal of the American Statistical '
                           'Association 44(247), 335–341, 1949.',
    'rfc8259': 'T. Bray (ed.), RFC 8259, The JavaScript Object Notation (JSON) Data Interchange Format, December 2017.',
    # the Isaac, LeRobot-config and report exporters (checks_formats.py: formats.isaac_*, lerobot_export, report)
    'openusd_camera': 'OpenUSD UsdGeomCamera (focalLength, horizontalAperture, verticalAperture and their offsets, '
                      'clippingRange; a camera looks along -Z with +Y up) and Gf.Camera / GfFrustum '
                      '(ComputeViewMatrix, ComputeProjectionMatrix), https://openusd.org/release/api/'
                      'class_usd_geom_camera.html, as implemented by usd-core 26.8 (pxr), read 2026-11-07.',
    'isaaclab_camera_v300': 'Isaac Lab tag v3.0.0 (commit e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4), '
                            'source/isaaclab/isaaclab/sensors/camera/camera_cfg.py (CameraCfg, OffsetCfg rot (x, y, '
                            'z, w), convention "ros"), tiled_camera_cfg.py (deprecated), camera.py '
                            '(_initialize_intrinsics), sim/spawners/sensors/sensors_cfg.py (PinholeCameraCfg, '
                            'OpenCvPinholeDistortionCfg, OpenCvFisheyeDistortionCfg) and sensors.py (spawn_camera, '
                            'omni:lensdistortion attributes), https://github.com/isaac-sim/IsaacLab, read 2026-11-07.',
    'isaaclab_camera_v232': 'Isaac Lab tag v2.3.2 (commit 37ddf626871758333d6ed89cf64ad702aef127d0), the same files: '
                            'CameraCfg.OffsetCfg rot (w, x, y, z), TiledCameraCfg, PinholeCameraCfg without OpenCV '
                            'distortion fields, https://github.com/isaac-sim/IsaacLab, read 2026-11-07.',
    'lerobot_train_config': 'Hugging Face LeRobot at commit d490e472a5d1a207cfe0255caaaf74b6d66e4656: '
                            'src/lerobot/configs/{train, default, policies, types}.py (TrainPipelineConfig, '
                            'DatasetConfig, PreTrainedConfig, FeatureType), policies/act/configuration_act.py, '
                            'policies/diffusion/configuration_diffusion.py, datasets/factory.py '
                            '(resolve_delta_timestamps), https://github.com/huggingface/lerobot, read 2026-11-07.',
    'act_repo': 'T. Z. Zhao et al., ACT, https://github.com/tonyzhaozh/act at commit '
                '742c753c0d4a5d87076c8f69e5628c79a8cc5488: constants.py (task configs: camera_names) and '
                'imitate_episodes.py (temporal aggregation, k = 0.01), read 2026-11-07.',
    'diffusion_policy_repo': 'C. Chi et al., Diffusion Policy, https://github.com/real-stanford/diffusion_policy at '
                             'commit 5ba07ac6661db573af695b419a7947ecb704690f: config/task/real_pusht_image.yaml '
                             '(shape_meta), config/train_diffusion_unet_real_image_workspace.yaml (n_latency_steps), '
                             'dataset/real_pusht_image_dataset.py, read 2026-11-07.',
    'openvla_oxe': 'OpenVLA, https://github.com/openvla/openvla at commit c8f03f48af692657d3060c19588038c7220e9af9: '
                   'prismatic/vla/datasets/rlds/oxe/configs.py (OXE_DATASET_CONFIGS: image_obs_keys, depth_obs_keys, '
                   'state_obs_keys, StateEncoding, ActionEncoding), read 2026-11-07.',
    'html_living_standard': 'WHATWG HTML Living Standard, https://html.spec.whatwg.org/, parsed here with the Python '
                            'standard library\'s html.parser (https://docs.python.org/3/library/html.parser.html).',
    'pdf_32000': 'ISO 32000-1:2008, Document management — Portable document format — Part 1: PDF 1.7 (§7.5 file '
                 'structure, §7.4.4 FlateDecode, §9.6.2.2 the standard 14 fonts); Adobe\'s public copy, '
                 'https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/PDF32000_2008.pdf.',
    'poppler_pdftotext': 'Poppler pdftotext, the PDF text extractor of the poppler-utils package, '
                         'https://poppler.freedesktop.org/.',
    # the robot catalogue (checks_robots.py)
    'mujoco_menagerie': 'MuJoCo Menagerie (Google DeepMind), a collection of MuJoCo models with per-model licences, '
                        'https://github.com/google-deepmind/mujoco_menagerie at commit '
                        '98a70c569c4c1b6da467a20012f766ee1f8dc521, read 2026-11-07.',
    'robot_datasheets': 'The makers\' datasheets and product pages each robot catalogue entry cites in its sources '
                        '(dexvantage/resources/library/robot/*.json), with the date each was read.',
    # project diff, the Git folder form, citations and figures (checks_versioning.py)
    'git_textconv': 'Git documentation, gitattributes(5), "Performing text diffs of binary files" (textconv: "The '
                    'program should take a single argument, the name of a file to convert, and produce the '
                    'resulting text on stdout"), https://git-scm.com/docs/gitattributes, read 2026-11-07.',
    'bibtex_btxdoc': 'BibTeX documentation btxdoc ("Using BibTeX" on CTAN), '
                     'https://mirrors.ctan.org/biblio/bibtex/base/btxdoc.pdf, listed at https://ctan.org/pkg/bibtex, '
                     'read 2026-11-07.',
    'svg11': 'W3C, Scalable Vector Graphics (SVG) 1.1 (Second Edition), W3C Recommendation 16 August 2011, '
             'https://www.w3.org/TR/SVG11/, read 2026-11-07.',
    'ieee_figure_size': 'IEEE Author Center, "Resolution and Size" ("One column width: 3.5 inches, 88.9 millimeters, '
                        'or 21 picas"; "Two columns width: 7.16 inches, 182 millimeters, or 43 picas"), '
                        'https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/'
                        'create-graphics-for-your-article/resolution-and-size/, read 2026-11-07.',
    'tol_colours': 'P. Tol, Paul Tol\'s Notes, "Qualitative colour schemes" (the bright scheme: #4477AA, #EE6677, '
                   '#228833, #CCBB44, #66CCEE, #AA3377, #BBBBBB), https://sronpersonalpages.nl/~pault/ (page dated 16 '
                   'August 2026), read 2026-11-07.',
    # library systems entries (checks_library_systems.py)
    'library_teleop_sources': 'The makers\' pages and the papers each teleop library entry cites in its sources '
                              '(dexvantage/resources/library/teleop/*.json), with the date each was read.',
    'library_policy_sources': 'The papers, model cards and official code each policy library entry cites in its '
                              'sources (dexvantage/resources/library/policy/*.json), with the date each was read.',
    'library_host_sources': 'The makers\' specification pages and datasheets each host library entry cites in its '
                            'sources (dexvantage/resources/library/host/*.json), with the date each was read.',
    'nist_sp811': 'NIST Special Publication 811, Guide for the Use of the International System of Units (SI), '
                  'Appendix B.8: pound (avoirdupois) = 4.535 924 E-01 kg (exactly 0.453 592 37 kg), '
                  'https://www.nist.gov/pml/special-publication-811, read 2026-11-07.',
    'dosovitskiy2021': 'A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at '
                       'Scale", ICLR 2021 (the ViT patch grid: (H/P)(W/P) patches).',
    # the ALOHA 2, bimanual Panda and G1 samples (checks_samples.py)
    'aloha2_2024': 'ALOHA 2 Team, "ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Teleoperation", 2024, '
                   'https://arxiv.org/html/2405.02292v1 (four RealSense D405 cameras at 848 x 480; the MuJoCo '
                   'Menagerie model "precisely" models all cameras and robot positions; read 2026-11-07).',
    'menagerie_98a70c5': 'MuJoCo Menagerie at commit 98a70c569c4c1b6da467a20012f766ee1f8dc521, folders aloha, '
                         'franka_emika_panda and unitree_g1, https://github.com/google-deepmind/mujoco_menagerie '
                         '(read 2026-11-07).',
    'unitree_g1_urdf': 'Unitree Robotics, unitree_ros robots/g1_description/g1_29dof_rev_1_0.urdf at commit '
                       '7c40519e02d7dd16c06b25fe3fa3b67fdeb7cd74, joint "d435_joint" (torso_link to d435_link: xyz '
                       '0.0576235 0.01753 0.42987 m, rpy 0 0.8307767239493009 0 rad), '
                       'https://github.com/unitreerobotics/unitree_ros (read 2026-11-07).',
}


def cite(keys) -> list:
    """[(key, text)] for the keys known (unknown ones are left out)."""
    return [(k, REFERENCES[k]) for k in keys if k in REFERENCES]
