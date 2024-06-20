from robot_sim.agents.robots.mobile_franka_panda import MobileFrankaPanda
from robot_sim.envs.basic_env import TaskBaseEnv
from robot_sim.utils import DataLog


class BasicAction:
    def __init__(self, env: TaskBaseEnv, max_steps: int = 50):
        self.env = env
        self.agent: MobileFrankaPanda = self.env.agent
        self.max_steps = max_steps
        self.datalog = DataLog()
        self.target_ctrl_mode = None
        self.check_success_input = {'success_fn': self.check_success}

    def run(self, *args, **kwargs):
        if self.target_ctrl_mode is not None:
            orig_ctrl_mode = self.agent.control_mode
            if self.target_ctrl_mode != orig_ctrl_mode:
                self.agent.set_control_mode(self.target_ctrl_mode)
        else:
            orig_ctrl_mode = None
        self.datalog.clear()
        ret = self.act(*args, **kwargs)
        if self.target_ctrl_mode is not None and self.target_ctrl_mode != orig_ctrl_mode:
            self.agent.set_control_mode(orig_ctrl_mode)
        return ret, self.datalog

    def get_goal_description(self):
        return 'Goal not set'

    def act(self, *args, **kwargs):
        action = kwargs.get('action', args[0])
        step_obs = []
        for _ in range(self.max_steps):
            obs = self.env.step(action)[0]
            action = None
            step_obs.append(obs)
            if self.env.render_mode == 'human':
                self.env.render_human()
            if self.success():
                self.datalog = self.datalog.from_list(step_obs)
                return True
        self.datalog = self.datalog.from_list(step_obs)
        return False

    def success(self):
        check_success_fn = self.check_success_input.get('success_fn', self.check_success)
        return check_success_fn()

    def check_success(self):
        raise NotImplementedError

    def __repr__(self):
        return self.get_goal_description()
