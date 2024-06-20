import os
import sys
import torch
import hydra
from omegaconf import OmegaConf
from mt_evaluator import MtEvaluator

CUR_DIR_PATH = os.path.dirname(__file__)
sys.path.append(CUR_DIR_PATH)
sys.path.append(os.path.abspath(os.path.join(CUR_DIR_PATH, '../..')))
sys.path = list(set(sys.path))

from IO_evaluator import test_loss


class MtEvaluatorSearch(MtEvaluator):
    def reset(self):
        self.task.reset()
        self.reset_obs()
        self.goal = 'search the %s from the objects on the table and transfer it to the basket' % self.task.target_obj.name
        self.lang_emb = torch.from_numpy(self.encoder.encode([self.goal])).to(self.device)
        print(self.goal)


@hydra.main('config', 'mt_evaluate_search', '1.3')
def main(conf: OmegaConf):
    task = hydra.utils.instantiate(conf.task)
    evaluator = MtEvaluatorSearch(task, conf.ckpt_dir)
    test_loss(
        ckpt_dir=conf.ckpt_dir,
        evaluator=evaluator,
        cam_views=evaluator.args['cam_view'],
        model_name=conf.model_name,
        proc_name=0,
        epoch=conf.model_name.split('-')[0],
        gpu_name='0',
        using_pos_and_orn=False,
        num_threads=0,
        test_num=conf.test_num
    )


if __name__ == "__main__":
    main()
