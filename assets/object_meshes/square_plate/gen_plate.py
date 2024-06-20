import numpy as np
from scipy.spatial.transform import Rotation as R

order = np.array([[2, 1, 4], [2, 4, 3],
                  [5, 6, 7], [5, 7, 8],
                  [8, 7, 3], [8, 3, 4],
                  [7, 6, 2], [7, 2, 3],
                  [6, 5, 1], [6, 1, 2],
                  [5, 8, 4], [5, 4, 1]])


def build_wall(points):
    vertices = order - 1
    normals = []
    for v in vertices:
        vn = np.cross(points[v[0]] - points[v[1]], points[v[2]] - points[v[1]])
        vn /= np.linalg.norm(vn)
        normals.append(vn)
    return points, normals

plate_height = 0.1
bottom_half_size = 0.15
top_half_size = 0.2
thickness = 0.001
bottom_v, bottom_n = build_wall(np.array([[-bottom_half_size, bottom_half_size, 0],
                                          [bottom_half_size, bottom_half_size, 0],
                                          [bottom_half_size, -bottom_half_size, 0],
                                          [-bottom_half_size, -bottom_half_size, 0],
                                          [-bottom_half_size, bottom_half_size, thickness],
                                          [bottom_half_size, bottom_half_size, thickness],
                                          [bottom_half_size, -bottom_half_size, thickness],
                                          [-bottom_half_size, -bottom_half_size, thickness]]))
wall1_v, wall1_n = build_wall(np.array([[-bottom_half_size, -bottom_half_size, thickness],
                                        [bottom_half_size, -bottom_half_size, thickness],
                                        [-bottom_half_size, -bottom_half_size, 0],
                                        [bottom_half_size, -bottom_half_size, 0],
                                        [-top_half_size, -top_half_size, plate_height],
                                        [top_half_size, -top_half_size, plate_height],
                                        [-(top_half_size + thickness), -(top_half_size + thickness), plate_height],
                                        [top_half_size + thickness, -(top_half_size + thickness), plate_height]]))

R1 = R.from_euler('xyz', [0, 0, 90], True).as_matrix()
wall2_v = np.matmul(wall1_v, R1.T)
wall2_n = np.matmul(wall1_n, R1.T)

R2 = R.from_euler('xyz', [0, 0, 180], True).as_matrix()
wall3_v = np.matmul(wall2_v, R2.T)
wall3_n = np.matmul(wall2_n, R2.T)

R3 = R.from_euler('xyz', [0, 0, -90], True).as_matrix()
wall4_v = np.matmul(wall3_v, R3.T)
wall4_n = np.matmul(wall3_n, R3.T)

output = ''
for points in [bottom_v, wall1_v, wall2_v, wall3_v, wall4_v]:
    for p in points:
        output += 'v %f %f %f\n' % (p[0], p[1], p[2])
for normals in [bottom_n, wall1_n, wall2_n, wall3_n, wall4_n]:
    for n in normals:
        output += 'vn %f %f %f\n' % (n[0], n[1], n[2])
for i in range(5):
    flag = 0
    for idx in order:
        if flag == 0:
            texture_order = [1, 2, 3]
        else:
            texture_order = [1, 3, 4]
        output += 'f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (idx[0] + i * 8, texture_order[0], idx[0] + i * 8,
                                                      idx[1] + i * 8, texture_order[1], idx[1] + i * 8,
                                                      idx[2] + i * 8, texture_order[2], idx[2] + i * 8)
        flag = 1 - flag
print(output)
