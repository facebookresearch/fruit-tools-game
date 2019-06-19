# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import torch.nn as nn
from torch.distributions import Gumbel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pdb
import numpy as np
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=2,
                        help='number of GPUs to use')
    parser.add_argument('--ncpu', type=int, default=2,
                        help='number of CPUs to use')
    parser.add_argument('--outf', default='.',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--modeldir', default='.',
                        help='folder to get model checkpoints for testing')
    parser.add_argument('--data_dir',help='folder to get data')
    parser.add_argument('--manualSeed', type=int,default=0,
                        help='manual seed')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=2)
    parser.add_argument('--variance',type=float,help='variance data sampling',
                                default=0.0)
    parser.add_argument('--gc',type=float,help='gradient clipping value',
                                default=0.1)
    parser.add_argument('--lr',type=float,help='learning rate',
                                default=0.001)
    parser.add_argument('--vocab_size',type=int,help='vocabulary size',
                                default=10)
    parser.add_argument('--max_length',type=int,help='max sentence length',
                                default=10)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--val_batch_size',type=int,default=100)
    parser.add_argument('--rnn_h_size',type=int,help='rnn hidden layer size',
                                default=100)
    parser.add_argument('--input_embedding_size',
                            help='input embedding size',type=int,default=100)
    parser.add_argument('--symb_embedding_size',
                            help='vocab symbols embedding size',
                                    type=int,default=50)
    parser.add_argument('--body_features_size',
                            help='body intermediate linear layer size',
                            type=int,default=256)
    parser.add_argument('--n_episodes',type=int,default=1500)
    parser.add_argument('--n_episodes_val',type=int,default=12)
    parser.add_argument('--T',help='max number of rounds in each episode',
                                        type=int,default=10)
    parser.add_argument('--norm_r',help='divide reward by max value achievable',
                                        type=int,default=2)
    parser.add_argument('--start_fruit',help='start with fruit',
                                        type=int,default=1)
    parser.add_argument('--min_r',help='cte added to reward',
                                        type=float,default=0.01)
    parser.add_argument('--block_mess',help='Messages channel is blocked',
                                        type=int,default=0)
    parser.add_argument('--history',help='Feed previous agent hidden state',
                                        type=int,default=0)
    parser.add_argument('--n_tools',type=int,default=2)
    parser.add_argument('--symmetric', help='switch input received',
                                        type=int, default=0)
    parser.add_argument('--corruptA', type=int, default=0)
    parser.add_argument('--corruptB', type=int, default=0)
    parser.add_argument('--sample_test', type=int, default=0)
    parser.add_argument('--st', help='use self-talk', type=int, default=0)
    opt = parser.parse_args()

    if opt.outf == '.':
        if os.environ.get('SLURM_JOB_DIR') is not None:
            opt.outf = os.environ.get('SLURM_JOB_DIR')

    if os.environ.get('SLURM_JOB_ID') is not None:
        opt.job_id = os.environ.get('SLURM_JOB_ID')

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    return opt

def return_bs_message(opt,batch_size,device):
    bs_message=torch.zeros(opt.max_length,batch_size,opt.vocab_size,device=device)
    probs=torch.ones(opt.vocab_size)/opt.vocab_size
    for l in range(opt.max_length):
        sample=torch.multinomial(probs,batch_size,replacement=True)
        bs_message[l,:,:]=one_hot(sample,opt.vocab_size)
    bs_message=bs_message.to(device)
    return bs_message

def one_hot(inds,depth):
    assert len(inds.size())==1
    device = inds.device
    y_onehot = torch.zeros((inds.size(0), depth), device=device)
    y_onehot[torch.arange(inds.size(0), device=device),inds] = 1
    y_onehot.requires_grad_()
    return y_onehot

def sample_stgs(proba_no_softmax, K, temp):
    device=proba_no_softmax.device
    dims=(proba_no_softmax.size(0),K)
    # sample with GS
    mean=torch.zeros(dims,device=device)
    scale=torch.ones(dims,device=device)
    samp_gumb=Gumbel(mean,scale).sample()
    g=samp_gumb.detach()
    out=(proba_no_softmax + g) / temp
    # soft approximation
    y=F.softmax(out,dim=-1)
    # use argmax at forward pass
    symb=y.argmax(dim=-1)
    # go throught one_hot for backward pass
    one_hot_symb=one_hot(symb,K)
    sample=(one_hot_symb - y).detach() + y
    return sample

def sample_val_batch(opt,fruits_samp3D,tools_samp3D,in_flabels,in_tlabels):

    idxF=np.zeros((2,opt.val_batch_size),dtype='int64')
    idxT1=np.zeros((2,opt.val_batch_size),dtype='int64')
    idxT2=np.zeros((2,opt.val_batch_size),dtype='int64')
    nsampf=fruits_samp3D.shape[1]
    nsampt=tools_samp3D.shape[1]
    for b in range(opt.val_batch_size):
        flabel=np.random.choice(in_flabels,1,replace=False)[0]
        f=np.random.choice(nsampf,1,replace=False)[0]
        tools_labels=np.random.choice(in_tlabels,2,replace=False)
        tlabel1=tools_labels[0]
        tlabel2=tools_labels[1]
        t1=np.random.choice(nsampt,1,replace=False)[0]
        t2=np.random.choice(nsampt,1,replace=False)[0]
        idxF[0,b]=flabel
        idxF[1,b]=f
        idxT1[0,b]=tlabel1
        idxT1[1,b]=t1
        idxT2[0,b]=tlabel2
        idxT2[1,b]=t2

    return idxF, idxT1, idxT2

def batch_artificial_diff(fruits_samp3D,fruits_names3D,fruits_labels3D,in_flabels,
                        tools_samp3D,tools_names3D,tools_labels3D,in_tlabels,
                        batch_size,n_tools):
    device=fruits_samp3D.device
    nsampf=fruits_samp3D.shape[1]
    ffeat_size=fruits_samp3D.shape[2]
    fruits_vectors=torch.zeros(batch_size,ffeat_size,device=device)
    flabels=torch.zeros(batch_size,dtype=torch.long,device=device)
    fnames=[]

    nsampt=tools_samp3D.shape[1]
    tfeat_size=tools_samp3D.shape[2]
    tools_vectors=torch.zeros(n_tools,batch_size,tfeat_size,device=device)
    tlabels=torch.zeros(n_tools,batch_size,dtype=torch.long,device=device)
    tnames1=[]
    tnames2=[]
    for b in range(batch_size):
        flabel=np.random.choice(in_flabels,1,replace=False)[0]
        f=np.random.choice(nsampf,1,replace=False)[0]
        fruits_vectors[b,:]=fruits_samp3D[flabel,f,:]
        flabels[b]=fruits_labels3D[flabel,f]
        fnames.append(fruits_names3D[flabel,f])

        tools_labels=np.random.choice(in_tlabels,2,replace=False)
        tlabel1=tools_labels[0]
        tlabel2=tools_labels[1]
        t1=np.random.choice(nsampt,1,replace=False)[0]
        t2=np.random.choice(nsampt,1,replace=False)[0]
        tools_vectors[0,b,:]=tools_samp3D[tlabel1,t1,:]
        tools_vectors[1,b,:]=tools_samp3D[tlabel2,t2,:]
        tlabels[0,b]=tools_labels3D[tlabel1,t1]
        tlabels[1,b]=tools_labels3D[tlabel2,t2]
        tnames1.append(tools_names3D[tlabel1,t1])
        tnames2.append(tools_names3D[tlabel2,t2])
    assert (tlabels[0,:]==tlabels[1,:]).sum()==0
    tnames1=np.array(tnames1)[np.newaxis].T
    tnames2=np.array(tnames2)[np.newaxis].T
    tnames=np.concatenate([tnames1,tnames2],axis=0).squeeze()
    fnames=np.array(fnames).squeeze(1)
    return fruits_vectors,flabels,fnames,tools_vectors,tlabels,tnames

class FruitsToolsDataset(object):
    def __init__(self,fruits_samp3D,fruits_names3D,fruits_labels3D,
                domain_flabels,tools_samp3D,tools_names3D,tools_labels3D,
                domain_tlabels,num_episodes,n_tools,bs):
        self.fruits_samp3D = fruits_samp3D
        self.fruits_names3D = fruits_names3D
        self.fruits_labels3D=fruits_labels3D
        self.tools_samp3D=tools_samp3D
        self.tools_names3D=tools_names3D
        self.tools_labels3D=tools_labels3D
        self.domain_flabels=domain_flabels
        self.domain_tlabels=domain_tlabels
        self.n_elements = num_episodes*bs
        self.n_tools = n_tools
        self.nsampf=self.fruits_samp3D.shape[1]
        self.nsampt=self.tools_samp3D.shape[1]

    def __len__(self):
        return self.n_elements

    def __getitem__(self, idx):
        device=self.fruits_samp3D.device

        idx_flabel=np.random.choice(self.domain_flabels,1,replace=True)[0]
        fsamp=np.random.choice(self.nsampf,1,replace=True)[0]
        fvectors=self.fruits_samp3D[idx_flabel,fsamp,:]
        flabels=self.fruits_labels3D[idx_flabel,fsamp,:]
        fnames=self.fruits_names3D[idx_flabel,fsamp,:].tolist()

        idx_tlabels=np.random.choice(self.domain_tlabels,2,replace=False)
        tsamps=np.random.choice(self.nsampt,2,replace=True)
        tvectors1=self.tools_samp3D[idx_tlabels[0],tsamps[0],:][None,:]
        tvectors2=self.tools_samp3D[idx_tlabels[1],tsamps[1],:][None,:]
        tvectors=torch.cat([tvectors1,tvectors2],dim=0)
        tlabels1=self.tools_labels3D[idx_tlabels[0],tsamps[0],:]
        tlabels2=self.tools_labels3D[idx_tlabels[1],tsamps[1],:]
        tlabels=torch.cat([tlabels1,tlabels2],dim=0)
        tnames1=self.tools_names3D[idx_tlabels[0],tsamps[0],:].tolist()
        tnames2=self.tools_names3D[idx_tlabels[1],tsamps[1],:].tolist()
        return fvectors,flabels,fnames,tvectors,tlabels,tnames1,tnames2

def get_max_rewards(opt,fruits_vectors,tool_vectors,tlabels,reward_function):

    device=fruits_vectors.device
    M_fruit=reward_function.M_fruit
    M_tool=reward_function.M_tool
    M=reward_function.M
    min_r=reward_function.min_r
    labels0=tlabels[0,:].type('torch.LongTensor')
    labels1=tlabels[1,:].type('torch.LongTensor')
    reward_mean0=reward_function.complete_reward_mean[labels0]
    reward_mean1=reward_function.complete_reward_mean[labels1]

    batch_size=fruits_vectors.size(0)
    sharing=torch.full((batch_size,1,1),0.5,device=device)
    tools0=tool_vectors.permute(1,2,0)[:,:,0]
    tools1=tool_vectors.permute(1,2,0)[:,:,1]
    prop_mat_fruits = torch.mm(fruits_vectors,M_fruit)
    prop_mat_tools0 = torch.mm(tools0,M_tool)
    prop_mat_tools1 = torch.mm(tools1,M_tool)

    Mtooldot0=torch.mm(M.t(),prop_mat_tools0.t()).unsqueeze(2)
    Mtooldot1=torch.mm(M.t(),prop_mat_tools1.t()).unsqueeze(2)
    rewards_both0=torch.bmm(prop_mat_fruits.unsqueeze(1),Mtooldot0.permute(1,0,2))
    rewards_both1=torch.bmm(prop_mat_fruits.unsqueeze(1),Mtooldot1.permute(1,0,2))
    rewards_both0+=min_r
    rewards_both1+=min_r
    rewards_both0/=reward_mean0
    rewards_both1/=reward_mean1
    rewards0=torch.bmm(sharing,rewards_both0).squeeze(2)
    rewards1=torch.bmm(sharing,rewards_both1).squeeze(2)
    all_rewards=torch.cat([rewards0,rewards1],dim=1)
    max_rewards,_=all_rewards.max(1)
    return max_rewards,all_rewards

def get_val_batch(fruits_samp3D,fruits_names3D,fruits_labels3D,
                        tools_samp3D,tools_names3D,tools_labels3D,
                        idxF,idxT1,idxT2):

    idx_flabels=idxF[0,:]
    idx_fsamples=idxF[1,:]
    idx_t1labels=idxT1[0,:]
    idx_t1samples=idxT1[1,:]
    idx_t2labels=idxT2[0,:]
    idx_t2samples=idxT2[1,:]

    fruits_vectors=fruits_samp3D[idx_flabels,idx_fsamples,:]
    flabels=fruits_labels3D[idx_flabels,idx_fsamples,:].squeeze(1)
    fnames=fruits_names3D[idx_flabels,idx_fsamples]

    tools_vectors_1=tools_samp3D[idx_t1labels,idx_t1samples,:]
    tools_vectors_2=tools_samp3D[idx_t2labels,idx_t2samples,:]
    tools_vectors_1=tools_vectors_1[None,:,:]
    tools_vectors_2=tools_vectors_2[None,:,:]
    tools_vectors=torch.cat([tools_vectors_1,tools_vectors_2],dim=0)

    labels1=tools_labels3D[idx_t1labels,idx_t1samples]
    labels2=tools_labels3D[idx_t2labels,idx_t2samples]
    labels1=labels1.squeeze(1)[None,:]
    labels2=labels2.squeeze(1)[None,:]
    tlabels=torch.cat([labels1,labels2],dim=0)

    names1=tools_names3D[idx_t1labels,idx_t1samples]
    names2=tools_names3D[idx_t2labels,idx_t2samples]
    names1=names1.squeeze(1)[None,:]
    names2=names2.squeeze(1)[None,:]
    tnames=np.concatenate([names1,names2],axis=0)
    assert (tlabels[0,:]==tlabels[1,:]).sum()==0
    return fruits_vectors,flabels,fnames,tools_vectors,tlabels,tnames

def masking_no_term(proposalA,proposalB,log_ppropA,log_ppropB,
                logpsentA,logpsentB,opt):
    batch_size=proposalB[0].shape[0]
    rangeb=range(batch_size)
    n_turns=max([len(proposalA),len(proposalB)])
    # there should not be any proposal at initialisation otherwise binary
    # starts at 1 and mask at 0
    binary=1.-proposalB[0][:,opt.n_choices].unsqueeze(1)
    mask_t=(1.-binary)
    pmask_t=mask_t.squeeze(1)
    # 0th step, all proposal are from B
    proposal=mask_t*proposalB[0]
    choice=proposalB[0][:,1:].argmax(-1)
    log_pprop=pmask_t*log_ppropB[0][rangeb,choice]
    log_psent=pmask_t*logpsentB[0] # already masked symbols
    for t in range(1,n_turns):
        ######## A
        proposal=mask_t*proposalA[t]+(1.-mask_t)*proposal
        choice=proposalA[t][:,1:].argmax(-1)
        log_pprop+=pmask_t*log_ppropA[t][rangeb,choice]
        # update termination
        binary=1.-proposalA[t][:,opt.n_choices].unsqueeze(1)
        mask_t=mask_t*(1.-binary) # those get fixed
        pmask_t=mask_t.squeeze(1)
        # gather message for the non-terminated
        log_psent+=pmask_t*logpsentA[t]
        ######## B
        proposal=mask_t*proposalB[t]+(1.-mask_t)*proposal
        choice=proposalB[t][:,1:].argmax(-1)
        log_pprop+=pmask_t*log_ppropB[t][rangeb,choice]
        # update termination
        binary=1.-proposalB[t][:,opt.n_choices].unsqueeze(1)
        mask_t=mask_t*(1.-binary) # those get fixed
        pmask_t=mask_t.squeeze(1)
        if t < (n_turns-1):
            # if n_turns=T+1, when t=n_turns-1=T (Tth step),
            # if some samples are not done there are non-zero pmask_t
            # but we don't consider the last message from B it is never read
            # if case n_turns < T+1, there all zero pmask_t anyway so this line is useless
            log_psent+=pmask_t*logpsentB[t]
    if n_turns < opt.T+1:
        assert pmask_t.eq(0).sum()==batch_size
    return proposal,log_pprop,log_psent

def return_stop_proba(log_pprop):
    log_pcont=log_pprop[:,-1]
    pcont=torch.exp(log_pcont)
    # for stability
    logpstop=torch.logsumexp(log_pprop[:,:-1],dim=-1)
    pstop=torch.exp(log_pprop)
    return pcont, log_pcont, pstop, logpstop

def masking_ent_prop(proposalA,proposalB,log_ppropA,log_ppropB,opt):
    n_turns=max([len(proposalA),len(proposalB)])
    # there should not be any proposal at initialisation otherwise binary
    # starts at 1 and mask at 0
    binary=1.-proposalB[0][:,opt.n_choices]
    mask_t=(1.-binary)
    # 0th step, all proposal are from B
    pcontB,log_pcontB,pstopB,logpstopB=return_stop_proba(log_ppropB[0])
    ent_prop=mask_t*(pcontB*log_pcontB + pstopB*logpstopB)
    for t in range(1,n_turns):
        ######## A
        pcontA,log_pcontA,pstopA,logpstopA=return_stop_proba(log_ppropA[t])
        ent_prop+=mask_t*(pcontA*log_pcontA + pstopA*logpstopA)
        # update termination
        binary=1.-proposalA[t][:,opt.n_choices]
        mask_t=mask_t*(1.-binary) # those get fixed
        ######## B
        pcontB,log_pcontB,pstopB,logpstopB=return_stop_proba(log_ppropB[t])
        ent_prop+=mask_t*(pcontB*log_pcontB + pstopB*logpstopB)
        # update termination
        binary=1.-proposalB[t][:,opt.n_choices]
        mask_t=mask_t*(1.-binary) # those get fixed
    return ent_prop

def masking_message(message,log_psymbols,EOS_token):
    batch_size=message.size(1)
    max_length=message.size(0)
    rangeb=range(batch_size)

    # MASK BY TERMINATION
    symbol=message[0,:,:].argmax(-1)
    log_psent=log_psymbols[0,rangeb,symbol]
    binary=message[0,:,EOS_token]
    mask_t=(1.-binary)
    for t in range(1,max_length):
        symbol=message[t,:,:].argmax(-1)
        log_psent+=mask_t*log_psymbols[t,rangeb,symbol]
        # update the mask, if it was EOS now set it to 0
        binary=message[t,:,EOS_token]
        mask_t=mask_t*(1.-binary)
    return log_psent

class RewardAgent(torch.nn.Module):
    def __init__(self,MappingMatrix,M_tool,M_fruit,agent_id,n_tools,n_choices,
                                            norm_r=False,min_r=0.1):
        super(RewardAgent, self).__init__()
        self.M=MappingMatrix
        self.M_tool=M_tool
        self.M_fruit=M_fruit
        self.agent_id=agent_id
        self.norm_r=norm_r
        self.min_r=min_r
        self.n_choices=n_choices
        self.n_tools=n_tools

    def normalise_rewards(self,fruits_properties,tools_properties,norm=True):

        prop_mat_fruits = torch.mm(fruits_properties,self.M_fruit)
        prop_mat_tools = torch.mm(tools_properties,self.M_tool)

        Mtooldot=torch.mm(self.M.t(),prop_mat_tools.t())
        rewards=torch.mm(prop_mat_fruits,Mtooldot)
        rewards+=self.min_r
        if norm:
            # use all data, but WARNING: new tools have no mean to use
            self.complete_reward_mean=rewards.mean(0)[:,None,None]
        else:
            self.complete_reward_mean=torch.ones_like(rewards.mean(0))[:,None,None]
        return rewards

    def compute_rewards(self,im_vectors,tool_vectors,tools_labels,
                            proposal,max_rewards):

        batch_size=im_vectors.shape[0]
        agent_share=proposal[:,0].unsqueeze(1).unsqueeze(2)
        tool_choice=proposal[:,1:self.n_tools+1]
        chosen_tools_vectors=torch.bmm(tool_vectors.permute(1,2,0),
                    tool_choice.unsqueeze(2)).squeeze(2)
        tools_idx=tool_choice[:,1].type('torch.LongTensor')
        labels=tools_labels.t()[range(batch_size),tools_idx].type('torch.LongTensor').detach()
        prop_mat_fruits = torch.mm(im_vectors,self.M_fruit)
        prop_mat_tools = torch.mm(chosen_tools_vectors,self.M_tool)

        Mtooldot=torch.mm(self.M.t(),prop_mat_tools.t()).unsqueeze(2)
        rewards_both=torch.bmm(prop_mat_fruits.unsqueeze(1),Mtooldot.permute(1,0,2))

        # NOTE: it is already added in the max_rewards
        rewards_both+=self.min_r
        rewards_both/=self.complete_reward_mean[labels,:,:]

        rewards=torch.bmm(agent_share,rewards_both)
        rewards=rewards.squeeze(2).squeeze(1)
        idx_pos=max_rewards.ne(0)
        idx_0=max_rewards.eq(0)
        normed_rewards=rewards.clone()
        normed_rewards[idx_pos]/=max_rewards[idx_pos]
        if idx_0.sum(0)>0.0:
            normed_rewards[idx_0]=1.
        if self.norm_r==1:
            rewards=normed_rewards
        # if normed_rewards is 1, then made the right choice, otherwise it's 0
        elif self.norm_r==2:
            idx_right=normed_rewards.eq(1.).type(torch.float)
            rewards=normed_rewards*idx_right

        # no termination proposal[:,opt.n_choices]=1
        agreed=1.-proposal[:,self.n_choices]
        #################### PENALIZE IF NOT TERMINATED
        rewards=rewards*agreed
        normed_rewards=normed_rewards*agreed
        return rewards,normed_rewards

class RewardAgentReinf(RewardAgent):
    def __init__(self,MappingMatrix,M_tool,M_fruit,agent_id,n_tools,n_choices,
                                            norm_r=False,min_r=0.1):
        super().__init__(MappingMatrix,M_tool,M_fruit,agent_id,n_tools,
                                            n_choices,norm_r,min_r)
        self.baseline=Baseline()
        self.baseline_loss = nn.MSELoss(reduce=False)

    def forward(self,im_vectors,tool_vectors,tools_labels,
                proposal,log_pprop,log_psent,max_rewards):

        rewards,normed_rewards=self.compute_rewards(im_vectors,tool_vectors,
                                            tools_labels,proposal,max_rewards)
        batch_size=im_vectors.size(0)

        if self.training:
            #################### REINFORCE if gradient is needed
            bsl=self.baseline(batch_size).squeeze(1)
            bsl_no_grad=bsl.detach().clone()
            rewards_no_grad=rewards.detach().clone()

            loss_baseline=self.baseline_loss(bsl,rewards_no_grad)
            loss_choice=-(rewards_no_grad - bsl_no_grad)*log_pprop
            loss_comm=-(rewards_no_grad - bsl_no_grad)*log_psent
            loss=(loss_choice+loss_comm)
            return loss,loss_baseline,rewards,normed_rewards.detach().clone()
        else:
            return rewards,normed_rewards.detach().clone()

class Baseline(nn.Module):

    def __init__(self, add_one=0):
        super(Baseline, self).__init__()
        self.bias = nn.Parameter(torch.ones(1))
    def forward(self, bs):
        batch_bias = (self.bias + 1.).expand(bs,1)
        return batch_bias

class PredictRandom(nn.Module):

    def __init__(self,n_tools):
        super(PredictRandom, self).__init__()
        self.n_tools=n_tools
    def predict(self,batch_size,device):

        prop_proba=torch.full((batch_size,self.n_tools),0.5,device=device)
        choice=torch.multinomial(prop_proba, 1).squeeze(1)
        # n_tools + 1 but we never sample continue with this predictor
        tool_choice=one_hot(choice,self.n_tools+1)
        share=torch.full((batch_size,1),1,device=device)
        proposal=torch.cat([share,tool_choice],dim=-1)
        return proposal

class PredictAverageReward(nn.Module):

    def __init__(self,M,M_tool,M_fruit,fruits_prop,tools_prop,min_r):
        super(PredictAverageReward, self).__init__()
        self.M=M
        self.M_tool=M_tool
        self.M_fruit=M_fruit
        self.fruits_prop=fruits_prop
        self.tools_prop=tools_prop
        self.min_r=min_r
        prop_mat_fruits = torch.mm(fruits_prop,self.M_fruit)
        prop_mat_tools = torch.mm(tools_prop,self.M_tool)
        Mtooldot=torch.mm(self.M.t(),prop_mat_tools.t())
        rewards=torch.mm(prop_mat_fruits,Mtooldot)

        rewards+=self.min_r
        self.rewards=rewards
        n_tools=tools_prop.shape[0]
        self.n_tools=n_tools

    def compute_overall_comp(self,domain_f,domain_t):
        # self.complete_rewards_mean=rewards.mean(0)
        # self.mean_rewards=rewards.mean(0)
        domain_rewards=self.rewards[domain_f,:]
        n_fruits=len(domain_f)
        overall_comp=torch.zeros(self.n_tools,self.n_tools)
        # for t1 in range(self.n_tools):
        for i,t1 in enumerate(domain_t):
            val1=domain_rewards[:,t1]
            for _,t2 in enumerate(domain_t[i:]):
                val2=domain_rewards[:,t2]
                if t1!=t2:
                    diff=((val1-val2)>=0).sum().item()/float(n_fruits)
                    overall_comp[t1,t2]=diff
                    overall_comp[t2,t1]=1.-diff
        self.overall_comp=overall_comp

    def predict(self,tools_labels):
        device=tools_labels.device
        batch_size=tools_labels.size(1)
        labels=tools_labels.type('torch.LongTensor')
        n_tools=tools_labels.size(0)
        choice=torch.zeros(batch_size,dtype=torch.long,device=device)
        for b in range(batch_size):
            t1=tools_labels[0,b]
            t2=tools_labels[1,b]
            comp=self.overall_comp[t1,t2]
            if comp>=0.5:
                choice[b]=0
            else:
                choice[b]=1
        # n_tools + 1 but we never sample continue with this predictor
        tool_choice=one_hot(choice,n_tools+1)
        share=torch.full((batch_size,1),1,device=device)
        proposal=torch.cat([share,tool_choice],dim=-1)
        return proposal
