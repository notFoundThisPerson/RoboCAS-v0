import json
import os
from collections import OrderedDict
import numpy as np
import torch
from gym import spaces
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
import util.misc as utils
from IO_trainer_torch import Trainer, load_config_from_json
from mt_dataset import build_dataset
from maruya24_rt1.tokenizers.utils import batched_space_sampler, np_to_tensor
from maruya24_rt1.transformer_network import TransformerNetwork
from maruya24_rt1.transformer_network_test_set_up import state_space_list
import time
from tqdm import tqdm


class MtTrainer(Trainer):
    def __init__(self, args):
        utils.set_seed()
        self.args = args
        self.args = utils.init_distributed_mode(self.args)
        self.checkpoint_dir, self.tensorboard_dir = self.make_log_dir(self.args["log_dir"])
        self.train_dataset, self.val_dataset = build_dataset(
            data_path=self.args["data_path"],
            time_sequence_length=self.args["time_sequence_length"],
            num_train_episode=self.args["num_train_episode"],
            num_val_episode=self.args["num_val_episode"],
            cam_view=self.args["cam_view"],
            **self.args.get('dataset_kwargs', dict())
        )

        if self.args["distributed"]:
            self.sampler_train = DistributedSampler(self.train_dataset, shuffle=True)
            self.sampler_val = DistributedSampler(self.val_dataset, shuffle=False)

        self.args["checkpoint_dir"] = self.checkpoint_dir
        self.writer_train = SummaryWriter(self.tensorboard_dir, flush_secs=5)
        self.writer_val = SummaryWriter(self.tensorboard_dir + "_val", flush_secs=5)
        with open(os.path.join(self.args["data_path"], 'data_info.json'), 'r') as f:
            info = json.load(f)
            action_space_limits = info['action_space']
        self._action_space = spaces.Dict(
            OrderedDict(
                [
                    ("terminate_episode", spaces.Discrete(4)),
                    (
                        "world_vector",
                        spaces.Box(
                            low=np.array(action_space_limits['relative_position'][0]),
                            high=np.array(action_space_limits['relative_position'][1]),
                            shape=(3,),
                            dtype=np.float32
                        ),
                    ),
                    (
                        "rotation_delta",
                        spaces.Box(
                            low=np.array(action_space_limits['relative_rotation'][0]),
                            high=np.array(action_space_limits['relative_rotation'][1]),
                            shape=(3,),
                            dtype=np.float32,
                        ),
                    ),
                    (
                        "gripper_closedness_action",
                        spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                    ),
                ]
            )
        )
        self.args["action_space"] = str(self._action_space)
        if utils.is_main_process():
            with open(os.path.join(self.checkpoint_dir, self.train_name + ".json"), "w") as json_file:
                json.dump(self.args, json_file)
            json_file.close()
        self.device = torch.device(self.args["device"])

        self.train_step = 0
        self.val_step = 0

    def calc_fk(self, obs):
        # device = obs['position'].device
        # batch_size, episode_length, _ = obs['position'].shape
        # obs['position'] = torch.cat(
        #     [obs['position'], torch.zeros([batch_size, episode_length, 1], dtype=torch.float32, device=device)], dim=-1)
        return obs

    def _initialize_network_cfg(self):
        # Initialize the TransformerNetwork based on specified configurations
        network_configs = self.args["network_configs"]
        # Modify network configuration based on specific settings
        network_configs["time_sequence_length"] = self.args["time_sequence_length"]
        network_configs["num_encoders"] = len(self.args["cam_view"])
        network_configs["token_embedding_size"] = network_configs["token_embedding_size_per_image"] * len(self.args["cam_view"])
        del network_configs["token_embedding_size_per_image"]
        network_configs["using_proprioception"] = self.args["using_proprioception"]
        network_configs["input_tensor_space"] = state_space_list()[0]
        network_configs["output_tensor_space"] = self._action_space
        return network_configs

    def train(self):
        print("training")
        # Create dataloader based on distributed or single-machine settings
        if self.args["distributed"]:
            # Batch sampler for distributed training
            batch_sampler_train = torch.utils.data.BatchSampler(
                self.sampler_train, self.args["batch_size"], drop_last=True
            )
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler_train,
                num_workers=self.args.get("num_workers", self.args["batch_size"]),
                pin_memory=True
            )
        else:
            # DataLoader for single-machine training
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args["batch_size"],
                num_workers=self.args.get("num_workers", 2),
                shuffle=True,
                drop_last=True,
                pin_memory=True
            )

        network_configs = self._initialize_network_cfg()
        network = TransformerNetwork(**network_configs)
        network.to(self.device)
        network_without_ddp = network

        # Load model weights, optimizer, scheduler settings, resume from checkpoints if specified
        total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print("number of model params:", total_params)
        total_size_bytes = total_params * 4
        # Parameter is in torch.float32，Each parameter takes 4 bytes
        total_size_mb = round(total_size_bytes / (1024 * 1024), 2)
        print("model size: ", total_size_mb, " MB")

        # Configuration based on distributed or single-machine setup
        if self.args["distributed"]:
            # DistributedDataParallel setup
            network = torch.nn.parallel.DistributedDataParallel(
                network, device_ids=[self.args["gpu"]], find_unused_parameters=False
            )
            network_without_ddp = network.module
            optimizer = torch.optim.AdamW(
                network_without_ddp.parameters(), lr=self.args["lr"]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer, **self.args["scheduler_configs"]
            )
            if self.args["resume"]:
                checkpoint = torch.load(self.args["resume_from_checkpoint"], map_location="cpu")
                network_without_ddp.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # Single-machine setup
            optimizer = torch.optim.AdamW(network.parameters(), lr=self.args["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer, **self.args["scheduler_configs"]
            )
            if self.args["resume"]:
                checkpoint = torch.load(self.args["resume_from_checkpoint"], map_location="cpu")
                network.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # self.val(network_without_ddp, 0, self.val_dataset)
        # Training loop over epochs
        epoch_start = checkpoint["epoch"] + 1 if self.args["resume"] else 0
        self.train_step = epoch_start * len(train_dataloader)
        for e in range(epoch_start, self.args["epochs"]):
            network.train()
            with tqdm(total=len(train_dataloader), dynamic_ncols=True, desc="train") as pbar:
                for i, (obs, action) in enumerate(train_dataloader):
                    # Perform training steps
                    optimizer.zero_grad()
                    network_without_ddp.set_actions(
                        utils.dict_to_device(action, self.device)
                    )
                    network_state = batched_space_sampler(
                        network_without_ddp._state_space,
                        batch_size=self.args["batch_size"],
                    )
                    network_state = np_to_tensor(network_state)
                    if self.args["using_proprioception"]:
                        obs = self.calc_fk(obs)
                    output_actions, network_state = network(
                        utils.dict_to_device(obs, self.device),
                        utils.dict_to_device(network_state, self.device),
                    )

                    loss = network_without_ddp.get_actor_loss().mean()

                    loss.backward()
                    total_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=4, norm_type=2)
                    optimizer.step()

                    # Logging metrics during training
                    if utils.is_main_process() and i % 50 == 0:
                        # Log loss, epoch, and learning rate
                        cur_time = time.time()
                        self.writer_train.add_scalar(
                            tag="loss_ce",
                            global_step=self.train_step,
                            scalar_value=loss.cpu().data.numpy(),
                            walltime=cur_time,
                        )
                        self.writer_train.add_scalar(
                            tag="epoch",
                            global_step=self.train_step,
                            scalar_value=e,
                            walltime=cur_time,
                        )
                        self.writer_train.add_scalar(
                            tag="lr",
                            global_step=self.train_step,
                            scalar_value=optimizer.state_dict()["param_groups"][0]["lr"],
                            walltime=cur_time,
                        )
                        self.writer_train.add_scalar(
                            tag='total_norm',
                            global_step=self.train_step,
                            scalar_value=total_norm.data,
                            walltime=cur_time
                        )
                        pbar.set_postfix(
                            ordered_dict={
                                "epoch": e,
                                # "train_name": self.train_name[-5:],
                                # "gpu_memory_used": str(round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)) + " GB",
                                "loss": loss.item(),
                                # "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                                "cur_time": time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
                            }
                        )
                        pbar.update(50)
                    self.train_step += 1

            # Perform validation at specified intervals
            if (e + 1) % self.args["val_interval"] == 0:
                checkpoint_filename = os.path.join(
                    self.checkpoint_dir, str(e) + "-checkpoint.pth"
                )
                checkpoint = {
                    "model_state_dict": network_without_ddp.state_dict()
                    if self.args["distributed"]
                    else network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "action_space": self._action_space,
                    "epoch": e,
                }
                utils.save_on_master(checkpoint, checkpoint_filename)
            scheduler.step()

    def validation(self):
        print("validating")

        # Initialize the TransformerNetwork based on specified configurations
        network_configs = self._initialize_network_cfg()
        network = TransformerNetwork(**network_configs)
        network.to(self.device)

        # Load model weights, optimizer, scheduler settings, resume from checkpoints if specified
        if self.args["resume"]:
            checkpoint = torch.load(self.args["resume_from_checkpoint"], map_location="cpu")
            network.load_state_dict(checkpoint["model_state_dict"])

        epoch_start = checkpoint["epoch"] + 1 if self.args["resume"] else 0
        self.val_step = epoch_start // self.args["val_interval"]

        if self.args["distributed"]:
            # Barrier synchronization for distributed training
            print(
                f"Process {torch.distributed.get_rank()} has reached the end of epoch {epoch_start}."
            )
            torch.distributed.barrier()
            self.val(
                network_without_ddp=network,
                epoch=epoch_start,
                val_dataset=self.val_dataset,
                sampler_val=self.sampler_val,
            )
            print(f"Process {torch.distributed.get_rank()} has reached the end of val.")
            torch.distributed.barrier()
        else:
            self.val(
                network_without_ddp=network,
                epoch=epoch_start,
                val_dataset=self.val_dataset,
            )


if __name__ == "__main__":
    args = load_config_from_json(os.path.join(os.path.dirname(__file__), "mt_train_config.json"))
    trainer = MtTrainer(args)
    if args["mode"] == "train":
        trainer.train()
    elif args["mode"] == "eval":
        trainer.evaluate()
    else:
        raise NotImplementedError("mode must be '''train''' or '''eval'''")
