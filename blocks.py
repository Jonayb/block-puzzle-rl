blocks = {
    0: {'name': 'single', 'coords': [[0, 0]],
        'matrix': [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    1: {'name': 'double', 'coords': [[0, 0], [1, 0]],
        'matrix': [[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    2: {'name': 'double_rotated', 'coords': [[0, 0], [0, 1]],
        'matrix': [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    3: {'name': 'triple', 'coords': [[0, 0], [1, 0], [2, 0]],
        'matrix': [[1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    4: {'name': 'triple_rotated', 'coords': [[0, 0], [0, 1], [0, 2]],
        'matrix': [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    5: {'name': 'quad', 'coords': [[0, 0], [1, 0], [2, 0], [3, 0]],
        'matrix': [[1, 1, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    6: {'name': 'quad_rotated', 'coords': [[0, 0], [0, 1], [0, 2], [0, 3]],
        'matrix': [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    7: {'name': 'five', 'coords': [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
        'matrix': [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    8: {'name': 'five_rotated', 'coords': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
        'matrix': [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]},
    9: {'name': 'square', 'coords': [[0, 0], [1, 0], [0, 1], [1, 1]],
        'matrix': [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    10: {'name': 'L', 'coords': [[0, 0], [0, 1], [0, 2], [1, 2]],
         'matrix': [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    11: {'name': 'L_rotated', 'coords': [[0, 0], [1, 0], [2, 0], [0, 1]],
         'matrix': [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    12: {'name': 'L_rotated_2', 'coords': [[0, 0], [1, 0], [1, 1], [1, 2]],
         'matrix': [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    13: {'name': 'L_rotated_3', 'coords': [[2, 0], [0, 1], [1, 1], [2, 1]],
         'matrix': [[0, 0, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    14: {'name': 'L_flipped', 'coords': [[1, 0], [1, 1], [0, 2], [1, 2]],
         'matrix': [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    15: {'name': 'L_flipped_rotated', 'coords': [[0, 0], [0, 1], [1, 1], [1, 2]],
         'matrix': [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    16: {'name': 'L_flipped_rotated_2', 'coords': [[0, 0], [1, 0], [0, 1], [0, 2]],
         'matrix': [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    17: {'name': 'L_flipped_rotated_3', 'coords': [[0, 0], [1, 0], [2, 0], [2, 1]],
         'matrix': [[1, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    18: {'name': 'Z', 'coords': [[0, 0], [0, 1], [1, 1], [1, 2]],
         'matrix': [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    19: {'name': 'Z_rotated', 'coords': [[1, 0], [2, 0], [0, 1], [1, 1]],
         'matrix': [[0, 1, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    20: {'name': 'Z_flipped', 'coords': [[1, 0], [0, 1], [1, 1], [0, 2]],
         'matrix': [[0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    21: {'name': 'Z_flipped_rotated', 'coords': [[0, 0], [1, 0], [1, 1], [2, 1]],
         'matrix': [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    22: {'name': 'T_small', 'coords': [[0, 0], [1, 0], [2, 0], [1, 1]],
         'matrix': [[1, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    23: {'name': 'T_small_rotated', 'coords': [[1, 0], [0, 1], [1, 1], [1, 2]],
         'matrix': [[0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    24: {'name': 'T_small_flipped', 'coords': [[1, 0], [0, 1], [1, 1], [2, 1]],
         'matrix': [[0, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    25: {'name': 'T_small_flipped_rotated', 'coords': [[0, 0], [0, 1], [1, 1], [0, 2]],
         'matrix': [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    26: {'name': 'T', 'coords': [[0, 0], [1, 0], [2, 0], [1, 1], [1, 2]],
         'matrix': [[1, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    27: {'name': 'T_rotated', 'coords': [[2, 0], [0, 1], [1, 1], [2, 1], [2, 2]],
         'matrix': [[0, 0, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    28: {'name': 'T_flipped', 'coords': [[1, 0], [1, 1], [0, 2], [1, 2], [2, 2]],
         'matrix': [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    29: {'name': 'T_flipped_rotated', 'coords': [[0, 0], [0, 1], [1, 1], [2, 1], [0, 2]],
         'matrix': [[1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    30: {'name': 'cross', 'coords': [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]],
         'matrix': [[0, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    31: {'name': 'angle', 'coords': [[0, 1], [1, 0], [1, 1]],
         'matrix': [[0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    32: {'name': 'angle_rotated', 'coords': [[0, 0], [1, 0], [1, 1]],
         'matrix': [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    33: {'name': 'angle_flipped', 'coords': [[0, 0], [0, 1], [1, 1]],
         'matrix': [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    34: {'name': 'angle_flipped_rotated', 'coords': [[0, 0], [0, 1], [1, 0]],
         'matrix': [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    35: {'name': 'big_angle', 'coords': [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]],
         'matrix': [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    36: {'name': 'big_angle_rotated', 'coords': [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]],
         'matrix': [[1, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    37: {'name': 'big_angle_flipped', 'coords': [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]],
         'matrix': [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    38: {'name': 'big_angle_flipped_rotated', 'coords': [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]],
         'matrix': [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    39: {'name': 'C', 'coords': [[0, 0], [0, 1], [1, 0], [2, 0], [2, 1]],
         'matrix': [[1, 1, 1, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    40: {'name': 'C_rotated', 'coords': [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2]],
         'matrix': [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    41: {'name': 'C_flipped', 'coords': [[0, 0], [0, 1], [1, 1], [2, 0], [2, 1]],
         'matrix': [[1, 0, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]},
    42: {'name': 'C_flipped_rotated', 'coords': [[0, 0], [0, 2], [1, 0], [1, 1], [1, 2]],
         'matrix': [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]}}