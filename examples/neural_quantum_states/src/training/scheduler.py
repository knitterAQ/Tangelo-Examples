import warnings
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler
import torch
import math
from typing import Callable

class VNAScheduler: # A custom scheduler class for scheduling the regularization parameter for variational neural annealing (VNA)
    def __init__(self, num_epochs: int, initial_constant: float=1.0, schedule_choice: str='none'):
        self.num_epochs = num_epochs
        self.initial_constant = initial_constant
        self.get_constant = self.get_scheduler(schedule_choice)

    def __call__(self, epoch: int) -> float: # Wrapper function of getting the regularization constant at 'epoch' iterations
        return self.get_constant(epoch)

    def polynomial_scheduler(self, epoch: int) -> float: # Annealing is polynomial with respect to training fraction
        train_frac = epoch/self.num_epochs
        if train_frac <=self.decay_start:
            return self.initial_constant
        elif train_frac > self.decay_end:
            return 0.0
        else:
            return ((train_frac - self.decay_end)/(self.decay_start - self.decay_end))**self.power

    def exponential_scheduler(self, epoch: int) -> float: # Annealing is exponential with respect to training fraction
        return self.initial_constant*self.decay_rate**epoch

    def cosine_scheduler(self, epoch: int) -> float: # Annealing decreases as a cosine wave with respect to training fraction
        eps = 0.0
        train_frac = epoch/self.num_epochs
        if train_frac <= self.decay_start:
            return self.initial_constant + eps
        elif train_frac >= self.decay_end:
            return eps
        else:
            return self.initial_constant*0.5*(1.0 + math.cos((math.pi/(self.decay_end - self.decay_start))*(train_frac - self.decay_start))) + eps
    
    def cosine_tent_scheduler(self, epoch: int) -> float: # Annealing increases and then decreases as a symmetric cosine curve
        train_frac = epoch/self.num_epochs
        if train_frac <= self.tent_start or train_frac >= self.tent_end:
            return 0.0
        else:
            return self.initial_constant*0.5*(1.0 - math.cos((2*math.pi/(self.tent_end - self.tent_start))*(train_frac - self.tent_start)))
    
    def trapezoid_scheduler(self, epoch: int) -> float: # Annealing increases linearly, holds constant, and then decreases linearly
        train_frac = epoch/self.num_epochs
        rate = min(1.0,(train_frac - self.up_start)/(self.up_end - self.up_start), (train_frac - self.down_end)/(self.down_end - self.down_start))
        return max(0.0,self.initial_constant*rate)
    
    def cutoff_scheduler(self, epoch: int) -> float: # Annealing holds constant and then vanishes instantly
        if epoch/self.num_epochs <= self.cutoff:
            return self.initial_constant
        else:
            return 0.0

    def get_scheduler(self, choice: str) -> Callable: # Sets scheduler function and assigns relevant attributes
        if choice in ['none']:
            self.cutoff = -1.0
            return self.cutoff_scheduler
        elif choice in ['const']:
            self.cutoff = 2.0
            return self.cutoff_scheduler
        elif choice in ['cutoff']:
            self.cutoff = 0.6
            return self.cutoff_scheduler
        elif choice in ['exponential']:
            self.decay_rate = 0.00001**(1/self.num_epochs)
            return self.exponential_scheduler
        elif choice in ['cosine']:
            self.decay_start = 0.05
            self.decay_end = 0.90
            return self.cosine_scheduler
        elif choice in ['cos_tent']:
            self.tent_start = 0.1
            self.tent_end = 0.9
            return self.cosine_tent_scheduler
        elif choice in ['trapezoid']:
            self.up_start = 0.1
            self.up_end = 0.4
            self.down_start = 0.6
            self.down_end = 0.8
            return self.trapezoid_scheduler
        elif choice in ['linear', 'quadratic', 'cubic', 'quartic', 'quintic']:
            self.decay_start = 0.05
            self.decay_end = 1.0
            if choice == 'linear':
                self.power = 1
            elif choice == 'quadratic':
                self.power = 2
            elif choice == 'cubic':
                self.power = 3
            elif choice == 'quartic':
                self.power = 4
            elif choice == 'quintic':
                self.power = 5
            return self.polynomial_scheduler

class TrapezoidLR(_LRScheduler):
    # Warm up before 1/4 epochs and cool down after 3/4 epochs
    def __init__(self, optimizer, milestones, last_epoch=-1):
        self.milestones = Counter(milestones)
        super(TrapezoidLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return [self.piecewise(self.last_epoch, base_lr) for base_lr in self.base_lrs]

    def piecewise(self, x, lr):
        milestones = list(sorted(self.milestones.elements()))
        # start with 1
        x = x + 1
        if x <= milestones[0]:
            return lr/milestones[0]*x
        elif (x <= milestones[1]) and (x > milestones[0]):
            return lr
        elif (x <= milestones[2]) and (x > milestones[1]):
            return lr*(milestones[2]-x)/(milestones[2]-milestones[1])
        else:
            return 0

# https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/tree/master
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/6b5e8953a80aef5b324104dc0c2e9b8c34d622bd/warmup_scheduler/scheduler.py#L5
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr for base_lr in self.base_lrs]
        return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def get_scheduler(sche_name: str, optimizer: torch.optim.Optimizer, learning_rate: float, epochs: int, min_rate: float=5e-6) -> _LRScheduler:
    if sche_name == 'decay':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs/7*3), int(epochs/7*5)], gamma=0.1)
        # warm up at the first 1/10 of total epochs
        scheduler = GradualWarmupScheduler(optimizer, int(epochs/10), scheduler)
    elif sche_name == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=int(epochs), max_lr=learning_rate, min_lr=5e-5, warmup_steps=0, gamma=1.0)
    elif sche_name == 'cosine_warmup':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=int(0.96*epochs), max_lr=learning_rate, min_lr=5e-8, warmup_steps=int(epochs*0.04), gamma=1.0)
    elif sche_name == 'cosine_warmrestart':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=int(epochs*0.1), cycle_mult=1.3310577386373, max_lr=learning_rate, min_lr=5e-5, warmup_steps=int(epochs*0.04), gamma=0.8)
    elif sche_name == 'cyclic':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=(learning_rate/100),
                                          max_lr=learning_rate, step_size_up=int(epochs*2/5),
                                          step_size_down=int(epochs*3/5), cycle_momentum=False)
    elif sche_name == 'trap':
        scheduler = TrapezoidLR(optimizer, milestones=[int(epochs/4), int(epochs*3/4), epochs])
    elif sche_name == 'const':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs/2)], gamma=1.0)
        # warm up for first 1% of total epochs
        scheduler = GradualWarmupScheduler(optimizer, int(0.01*epochs), scheduler)
    else:
        raise "sche_name not recognized."
    return scheduler
