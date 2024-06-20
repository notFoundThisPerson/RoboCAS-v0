import hydra
import os
from omegaconf import OmegaConf


@hydra.main(os.path.join(os.path.dirname(__file__), '../config'), 'grasp_and_convey_table_random', '1.3')
def main(conf: OmegaConf):
    task = hydra.utils.instantiate(conf.task)
    task.run(conf.num_episodes)


if __name__ == '__main__':
    main()
