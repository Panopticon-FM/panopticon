# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from collections import defaultdict, deque
import datetime
import json
import logging
import time

import torch
import numpy as np
import dinov2.distributed as distributed
import wandb 
import os
import math

logger = logging.getLogger("dinov2")


class MetricLogger(object):
    def __init__(self, delimiter="\t", output_dir=None, output_file=None, use_wandb=False):
        self.meters = defaultdict(SmoothedValue)
        self.expert_meters = {}  # Separate dict for expert metrics
        self.delimiter = delimiter
        self.output_file = os.path.join(output_dir, output_file) if output_dir else output_file
        self.use_wandb = use_wandb
        self.epoch_len = None
        self.nsamples_per_iter = None
        self.dataset_len = None
        # Track which metrics should be logged as bar charts
        self.expert_metrics = set()

        if use_wandb:
            assert wandb.run is not None, 'wandb.run needs to be initialized before MetricLogger'
            self.run = wandb.run
        self.iter_time = SmoothedValue(fmt="{avg:.6f}")
        self.data_time = SmoothedValue(fmt="{avg:.6f}")
        self.epoch_time = SmoothedValue(fmt="{avg:.6f}")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update_expert_metrics(self, name, values):
        """Update expert-specific metrics for bar chart visualization.
        Only stores the most recent values - no smoothing applied.
        These values will persist until the next update.
        """
        for i, v in enumerate(values):
            if isinstance(v, (torch.Tensor, np.ndarray)):
                v = float(v)
            self.expert_meters[f"{name}/expert_{i}"] = v

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not distributed.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump['epoch'] = iteration // self.epoch_len
        # Regular metrics use median
        dict_to_dump.update({k: float(v.median) for k, v in self.meters.items()})
        # Expert metrics - ensure all values are Python floats
        dict_to_dump.update({k: float(v) for k, v in self.expert_meters.items()})
    
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

        if self.use_wandb:
            # improved ordering for wandb online GUI
            pattern = {
                'params': ['lr','wd','mom','teacher_temp', 'repa_alpha'],
                'loss': ['total_loss','dino_local_crops_loss','dino_global_crops_loss','koleo_loss','ibot_loss','aux_loss','pe_distil_loss'],
                'expert_activations': [k for k in self.expert_meters.keys()]
            }
            for pat, v in pattern.items():
                to_log = {}
                for k in v:
                    if k in dict_to_dump:
                        to_log[k] = dict_to_dump.pop(k)
                self.log_wandb(iteration, to_log, prefix=pat)
            if len(dict_to_dump) > 0:
                self.log_wandb(iteration, dict_to_dump)

    def log_wandb(self, iteration, metric_dict, prefix=None, log_step=True):
        if not self.use_wandb:
            return
        if prefix:
            metric_dict = {os.path.join(prefix,str(k)): v for k, v in metric_dict.items()}

        # define progress values
        metric_dict['iteration'] = iteration
        metric_dict['epoch'] = iteration // self.epoch_len
        if self.nsamples_per_iter is not None:
            metric_dict['nsamples'] = iteration * self.nsamples_per_iter
            if self.dataset_len is not None:
                metric_dict['epoch_act'] = (iteration * self.nsamples_per_iter) // self.dataset_len

        if log_step:
            self.run.log(metric_dict, step=iteration)
        else:
            self.run.log(metric_dict)
    

    def log_every(self, 
                  iterable, 
                  print_freq, 
                  header=None, 
                  n_iterations=None, 
                  start_iteration=0, 
                  use_self_timemeters=False, 

                  # kwargs for logging progress from different viewpoints
                  nsamples_per_iter=None,
                  dataset_len=None,
                  epoch_len=None):
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        epoch_start = time.time()
        iter_time = self.iter_time if use_self_timemeters else SmoothedValue(fmt="{avg:.6f}")
        data_time = self.data_time if use_self_timemeters else SmoothedValue(fmt="{avg:.6f}")
        epoch_time = self.epoch_time if use_self_timemeters else SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)
        epoch_len = epoch_len or n_iterations
        self.epoch_len = epoch_len
        self.nsamples_per_iter = nsamples_per_iter
        self.dataset_len = dataset_len
        n_epochs = math.ceil(n_iterations / epoch_len)

        i = start_iteration
        epoch = int(i // epoch_len)

        iter_space_fmt = ":" + str(len(str(n_iterations))) + "d"
        epoch_space_fmt = ":" + str(len(str(n_epochs))) + "d"

        log_list = [
            header,
            "[iter: {0" + iter_space_fmt + "}/{1}, ",
            "epoch: {2" + epoch_space_fmt + "}/{3}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                self.dump_in_output_file(iteration=i, iter_time=iter_time.avg, data_time=data_time.avg)
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            epoch,
                            n_epochs,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            if i // epoch_len > epoch: # this allows float epoch_len
                epoch_time_current = time.time() - epoch_start
                logger.info(f"Epoch {epoch}/{n_epochs} done in {epoch_time_current:.2f}s\n")
                epoch_start = time.time()
                epoch += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} Total time: {} ({:.6f} s / it)\n".format(header, total_time_str, total_time / n_iterations))


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self):
        """
        Distributed synchronization of the metric
        Warning: does not synchronize the deque!
        """
        if not distributed.is_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
