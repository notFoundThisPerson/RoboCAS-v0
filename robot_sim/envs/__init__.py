import sys
import pathlib

CUR_DIR_PATH = pathlib.Path.cwd()
mani_skill2_path = CUR_DIR_PATH.parents[1] / '3rd_party/ManiSkill'
sys.path.append(mani_skill2_path.as_posix())
sys.path = list(set(sys.path))
