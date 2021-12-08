import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import implicit_maml.utils as utils
import datetime
from collections import deque
from logger import Logger
from tensorboardX import SummaryWriter
import os
import time
from   torch.nn import functional as F
from make_envs import make_env
from agent import make_agent
from utils import eval_mode, get_concat_samples, evaluate, soft_update, hard_update
from memory import Memory
from logger import Logger
import hydra
import types
from train_iq import irl_update, save, get_buffers, irl_update_inner, ilr_update_critic2, ilr_update_critic, iq_learn_update2
from wrappers.atari_wrapper import LazyFrames
import wandb



class Learner:
    def __init__(self, model, loss_function, inner_lr=1e-3, outer_lr=1e-2, GPU=False, inner_alg='gradient', outer_alg='adam'):
        self.model = model
        self.use_gpu = GPU
        #if GPU:
        #    self.model.cuda()
        assert outer_alg == 'sgd' or 'adam'
    
        #assert inner_alg == 'gradient' # sqp unsupported in this version
        self.inner_alg = inner_alg

    def get_params(self):
        return torch.cat([param.data.view(-1) for param in self.model.parameters()], 0).clone()

    def set_params(self, param_vals):
        offset = 0
        for param in self.model.parameters():
            param.data.copy_(param_vals[offset:offset + param.nelement()].view(param.size()))
            offset += param.nelement()
            
    def set_outer_lr(self, lr):
        for param_group in self.outer_opt.param_groups:
            param_group['lr'] = lr
            
    def set_inner_lr(self, lr):
        for param_group in self.inner_opt.param_groups:
            param_group['lr'] = lr

    def regularization_loss(self, w_0, lam=0.0):
        """
        Add a regularization loss onto the weights
        The proximal term regularizes around the point w_0
        Strength of regularization is lambda
        lambda can either be scalar (type float) or ndarray (numpy.ndarray)
        """
        regu_loss = 0.0
        offset = 0
        regu_lam = lam if type(lam) == float or np.float64 else utils.to_tensor(lam)
        if w_0.dtype == torch.float16:
            try:
                regu_lam = regu_lam.half()
            except:
                regu_lam = np.float16(regu_lam)
        for param in self.model.parameters():
            delta = param.view(-1) - w_0[offset:offset + param.nelement()].view(-1)
            if type(regu_lam) == float or np.float64:
                regu_loss += 0.5 * regu_lam * torch.sum(delta ** 2)
            else:
                # import ipdb; ipdb.set_trace()
                param_lam = regu_lam[offset:offset + param.nelement()].view(-1)
                param_delta = delta * param_lam
                regu_loss += 0.5 * torch.sum(param_delta ** 2)
            offset += param.nelement()
        return regu_loss

    def get_loss(self, args, draft):
        GAIL(expert.dataset(draft))
        return critic_loss, actor_losses, alpha_losses

    def predict(self, x, return_numpy=False):
        yhat = self.model.forward(utils.to_device(x, self.use_gpu))
        if return_numpy:
            yhat = utils.to_numpy(yhat)
        return yhat

    def learn_on_data(self, env_args, env,eval_env,args,expert_memory_replay, num_steps=10,
                      add_regularization=False,
                      w_0=None, lam=0.0):
        for epoch in range(1, n_epochs+1):
        # update policy n_iter times
        policy.update(n_iter, batch_size)
        
        # evaluate in environment
        total_reward = 0
        for episode in range(n_eval_episodes):
            state = env.reset()
            for t in range(max_timesteps):
                action = policy.select_action(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
                
        avg_reward = int(total_reward/n_eval_episodes)
        print("Epoch: {}\tAvg Reward: {}".format(epoch, avg_reward))
        
        # add data for graph
        epochs.append(epoch)
        rewards.append(avg_reward)
        
        if avg_reward > solved_reward:
            print("########### Solved! ###########")
            policy.save(directory, filename)

    def learn_task(self, expert_memory_replay, env_args, env, eval_env, args, num_steps, add_regularization=False, w_0=None, lam=0.0):
        REPLAY_MEMORY = int(env_args.replay_mem)
        return self.learn_on_data(env_args, env,eval_env,args,expert_memory_replay, num_steps,
                      add_regularization,
                      w_0, lam)

    def move_toward_target(self, target, lam=2.0):
        """
        Move slowly towards the target parameter value
        Default value for lam assumes learning rate determined by optimizer
        Useful for implementing Reptile
        """
        # we can implement this with the regularization loss, but regularize around the target point
        # and with specific choice of lam=2.0 to preserve the learning rate of inner_opt
        self.outer_opt.zero_grad()
        loss = self.regularization_loss(target, lam=lam)
        loss.backward()
        self.outer_opt.step()



    def matrix_evaluator(self, loss_type, env,args):
        """
        Constructor function that can be given to CG optimizer
        Works for both type(lam) == float and type(lam) == np.ndarray
        """
        if type(lam) == np.ndarray:
            lam = utils.to_device(lam, self.use_gpu)
        def evaluator(v):
            hvp = self.hessian_vector_product(task,loss_type)
            Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping)
            return Av
        return evaluator

    def hessian_vector_product(self, task, loss_type, env, args,expert_memory_replay, EPISODE_STEPS, REPLAY_MEMORY, INITIAL_MEMORY, fast_learner, vector, x=None, y=None):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        if x is not None and y is not None: 
        else:
            hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat

    
def make_GAIL_network(args):
        env = make_env('LeageDraftEnv')
        model = GAIL(env_name, state_dim, action_dim, max_action, lr, betas)
    return model


