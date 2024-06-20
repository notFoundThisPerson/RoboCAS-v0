from robot_sim.utils import format_path
import os
from random import sample

mtl_template = '''
newmtl {mtl_name}
illum 3
d 1
Ns 30
Kd 1 1 1 
Ks 0.06666666666666667 0.06666666666666667 0.06666666666666667 
map_Kd {texture_path}
'''


def gen_room_obj(x, y, z, random=True):
    xmin, xmax = -x / 2, x / 2
    ymin, ymax = -y / 2, y / 2

    global mtl_idx, v_cnt, vn_cnt, mtl_info, obj_info
    v_cnt = 0
    vn_cnt = 0
    mtl_idx = 0

    obj_info = 'mtllib rand_background.obj.mtl\nvt 0 0 0\nvt 1 0 0 \nvt 1 1 0\nvt 0 1 0\n'
    mtl_info = ''

    def add_wall(p1, p2, p3, p4, norm, mtl_name):
        plane_obj = 'usemtl %s\n' % mtl_name
        for p in [p1, p2, p3, p4]:
            plane_obj += 'v %f %f %f\n' % (p[0], p[1], p[2])
        plane_obj += 'vn %f %f %f\n' % (norm[0], norm[1], norm[2])
        global v_cnt, vn_cnt, obj_info
        plane_obj += 'f {0:d}/1/{3:d} {1:d}/2/{3:d} {2:d}/3/{3:d}\n'.format(v_cnt + 1, v_cnt + 2, v_cnt + 3, vn_cnt + 1)
        plane_obj += 'f {0:d}/1/{3:d} {1:d}/3/{3:d} {2:d}/4/{3:d}\n'.format(v_cnt + 1, v_cnt + 3, v_cnt + 4, vn_cnt + 1)
        v_cnt += 4
        vn_cnt += 1
        obj_info += plane_obj

    def add_texture(file_name):
        global mtl_idx, mtl_info
        mtl_idx += 1
        mtl_name = 'material_%d' % mtl_idx
        mtl_info += mtl_template.format(mtl_name=mtl_name, texture_path=file_name) + '\n'
        return mtl_name

    root_path = format_path('{MT_ASSET_DIR}/textures')
    floor_textures = os.listdir(os.path.join(root_path, 'ground_textures'))
    wall_textures = os.listdir(os.path.join(root_path, 'wall_textures'))

    if random:
        mtl_names = []
        for _ in range(4):
            mtl_names.append(add_texture(os.path.join('wall_textures', sample(wall_textures, 1)[0])))
        floor_mtl_name = add_texture(os.path.join('ground_textures', sample(floor_textures, 1)[0]))
    else:
        mtl_names = [add_texture('wall_textures/default_wall.jpg')] * 4
        floor_mtl_name = add_texture('ground_textures/default_floor.jpg')

    add_wall([xmin, ymax, 0], [xmax, ymax, 0], [xmax, ymax, z], [xmin, ymax, z], [0, -1, 0], mtl_names[0])
    add_wall([xmax, ymax, 0], [xmax, ymin, 0], [xmax, ymin, z], [xmax, ymax, z], [-1, 0, 0], mtl_names[1])
    add_wall([xmax, ymin, 0], [xmin, ymin, 0], [xmin, ymin, z], [xmax, ymin, z], [0, 1, 0], mtl_names[2])
    add_wall([xmin, ymin, 0], [xmin, ymax, 0], [xmin, ymax, z], [xmin, ymin, z], [1, 0, 0], mtl_names[3])
    mtl_name = add_texture('sky.jpg')
    add_wall([xmin, ymin, z], [xmin, ymax, z], [xmax, ymax, z], [xmax, ymin, z], [0, 0, -1], mtl_name)

    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            add_wall([x, y, 0], [min(x + 1, xmax), y, 0], [min(x + 1, xmax), min(y + 1, ymax), 0], [x, min(y + 1, ymax), 0],
                     [0, 0, 1], floor_mtl_name)
            y += 1
        x += 1

    with open(os.path.join(root_path, 'rand_background.obj'), 'w') as f:
        f.write(obj_info)
    with open(os.path.join(root_path, 'rand_background.obj.mtl'), 'w') as f:
        f.write(mtl_info)


if __name__ == '__main__':
    gen_room_obj(5, 5, 3)
