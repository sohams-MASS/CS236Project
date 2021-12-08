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
        #self.model.actor.inner_opt = torch.optim.SGD(self.model.parameters(), lr=inner_lr)
        #self.model.critic.inner_opt = torch.optim.SGD(self.model.parameters(), lr=inner_lr)
        #if outer_alg == 'adam':
        #    self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=outer_lr, eps=1e-3)
        #else:
        #    self.outer_opt = torch.optim.SGD(self.model.parameters(), lr=outer_lr)
        #self.loss_function = loss_function
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
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config["horizon"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]

        opt_d = torch.optim.Adam(self.d.parameters())

        exp_rwd_iter = []

        exp_obs = []
        exp_acts = []

        steps = 0
        while steps < num_steps_per_iter:
            ep_obs = []
            ep_rwds = []

            t = 0
            done = False

            ob = env.reset()

            while not done and steps < num_steps_per_iter:
                act = expert.act(ob)

                ep_obs.append(ob)
                exp_obs.append(ob)
                exp_acts.append(act)

                if render:
                    env.render()
                ob, rwd, done, info = env.step(act)

                ep_rwds.append(rwd)

                t += 1
                steps += 1

                if horizon is not None:
                    if t >= horizon:
                        done = True
                        break

            if done:
                exp_rwd_iter.append(np.sum(ep_rwds))

            ep_obs = FloatTensor(np.array(ep_obs))
            ep_rwds = FloatTensor(ep_rwds)

        exp_rwd_mean = np.mean(exp_rwd_iter)
        print(
            "Expert Reward Mean: {}".format(exp_rwd_mean)
        )

        exp_obs = FloatTensor(np.array(exp_obs))
        exp_acts = FloatTensor(np.array(exp_acts))

        rwd_iter_means = []
        for i in range(num_iters):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            while steps < num_steps_per_iter:
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []

                t = 0
                done = False

                ob = env.reset()

                while not done and steps < num_steps_per_iter:
                    act = self.act(ob)

                    ep_obs.append(ob)
                    obs.append(ob)

                    ep_acts.append(act)
                    acts.append(act)

                    if render:
                        env.render()
                    ob, rwd, done, info = env.step(act)

                    ep_rwds.append(rwd)
                    ep_gms.append(gae_gamma ** t)
                    ep_lmbs.append(gae_lambda ** t)

                    t += 1
                    steps += 1

                    if horizon is not None:
                        if t >= horizon:
                            done = True
                            break

                if done:
                    rwd_iter.append(np.sum(ep_rwds))

                ep_obs = FloatTensor(np.array(ep_obs))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
                # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts))\
                    .squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_costs[i:]) for i in range(t)]
                )
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))
                ).detach()
                ep_deltas = ep_costs.unsqueeze(-1)\
                    + gae_gamma * next_vals\
                    - curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)

                gms.append(ep_gms)

            rwd_iter_means.append(np.mean(rwd_iter))
            print(
                "Iterations: {},   Reward Mean: {}"
                .format(i + 1, np.mean(rwd_iter))
            )

            obs = FloatTensor(np.array(obs))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()

            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)

            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores)
                )
            loss.backward()
            opt_d.step()

            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v)\
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.v, new_params)

            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)

                return (advs * torch.exp(
                            distb.log_prob(acts)
                            - old_distb.log_prob(acts).detach()
                        )).mean()

            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()

                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * (
                            (old_cov / cov).sum(-1)
                            + (((old_mean - mean) ** 2) / cov).sum(-1)
                            - self.action_dim
                            + torch.log(cov).sum(-1)
                            - torch.log(old_cov).sum(-1)
                        ).mean()

            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()

                return hessian + cg_damping * v

            g = get_flat_grads(L(), self.pi).detach()

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )

            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts))\
                .mean()
            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.pi
            )
            new_params += lambda_ * grad_disc_causal_entropy

            set_params(self.pi, new_params)

        return exp_rwd_mean, rwd_iter_means
        
                
  


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



    def matrix_evaluator(self, loss_type, env,args, expert_memory_replay, EPISODE_STEPS, REPLAY_MEMORY, INITIAL_MEMORY, fast_learner,task, lam, regu_coef=1.0, lam_damping=10.0, x=None, y=None):
        """
        Constructor function that can be given to CG optimizer
        Works for both type(lam) == float and type(lam) == np.ndarray
        """
        if type(lam) == np.ndarray:
            lam = utils.to_device(lam, self.use_gpu)
        def evaluator(v):
            hvp = self.hessian_vector_product(task,loss_type, env, args, expert_memory_replay, EPISODE_STEPS, REPLAY_MEMORY, INITIAL_MEMORY, fast_learner, v, x=x, y=y)
            Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping)
            return Av
        return evaluator

    def hessian_vector_product(self, task, loss_type, env, args,expert_memory_replay, EPISODE_STEPS, REPLAY_MEMORY, INITIAL_MEMORY, fast_learner, vector, x=None, y=None):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        if x is not None and y is not None:
            online_memory_replay = get_buffers(env, args,EPISODE_STEPS, REPLAY_MEMORY, INITIAL_MEMORY, fast_learner)    
        else:
            online_memory_replay = get_buffers(env, args,EPISODE_STEPS, REPLAY_MEMORY, INITIAL_MEMORY, fast_learner)
        policy_batch = online_memory_replay.get_samples(args.train.batch, args.device)
        expert_batch = expert_memory_replay.get_samples(args.train.batch, args.device)
        critic_loss, actor_loss, alpha_loss = self.get_loss(args, policy_batch, expert_batch)
        if loss_type == 'critic':
            tloss = critic_loss
            grad_ft = torch.autograd.grad(tloss, self.model.critic.parameters(), create_graph=True)
            flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
            vec = utils.to_device(vector, self.use_gpu)
            h = torch.sum(flat_grad * vec)
            hvp = torch.autograd.grad(h, self.model.critic.parameters(),retain_graph=True)
            hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])

        elif loss_type == 'actor':
            tloss = actor_loss
            grad_ft = torch.autograd.grad(tloss, self.model.actor.parameters(), create_graph=True)
            flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
            vec = utils.to_device(vector, self.use_gpu)
            h = torch.sum(flat_grad * vec)
            hvp = torch.autograd.grad(h, self.model.actor.parameters(),retain_graph=True)
            hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        else:
            tloss = alpha_loss
            grad_ft = torch.autograd.grad(alpha_loss, self.model.log_alpha, create_graph = True)
            flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
            vec = utils.to_device(vector, self.use_gpu)
            h = torch.sum(flat_grad * vec)
            hvp = torch.autograd.grad(h, self.model.log_alpha,retain_graph=True)
            hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat


def make_fc_network(in_dim=1, out_dim=1, hidden_sizes=(40,40), float16=False):
    non_linearity = nn.ReLU()
    model = nn.Sequential()
    model.add_module('fc_0', nn.Linear(in_dim, hidden_sizes[0]))
    model.add_module('nl_0', non_linearity)
    model.add_module('fc_1', nn.Linear(hidden_sizes[0], hidden_sizes[1]))
    model.add_module('nl_1', non_linearity)
    model.add_module('fc_2', nn.Linear(hidden_sizes[1], out_dim))
    if float16:
        return model.half()
    else:
        return model

    
def make_SAC_network(args, task='PickPlaceMetaWorld'):
    assert task == 'PickPlaceMetaWorld'
    
    if task == 'PickPlaceMetaWorld':
        env = make_env(args)
        model= make_agent(env, args)
    
    return model


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

    
def model_imagenet_arch(in_channels, out_dim, num_filters=32, batch_norm=True, bias=True):
    raise NotImplementedError

