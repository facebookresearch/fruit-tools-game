# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
import torch.utils.data
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from architectures import *
import torch.optim as optim
import sys
import pdb
import datetime
import pickle
from utils import *
import time

def gradClamp(parameters, clip=1):

    for p in parameters:
        if p.grad is not None:
            p.grad.clamp_(min=-clip)
            p.grad.clamp_(max=clip)

def count_done(proposalA,n_choices,done):
    # it's terminated if a choice is made, e.g. null is 0
    idx=(1-proposalA[:,n_choices]).nonzero()
    if idx.numel()>0:
        idx=idx.squeeze(1)
    idx=idx.tolist()
    res=list(set(idx).difference(done))
    done.extend(res)
    count=len(res)
    return count

def initialise_output_fixed_message(opt,batch_size,device):
    prop=torch.zeros(batch_size,opt.n_choices+1,device=device)
    prop[:,0]=0.5 #sharing

    tool_choice_init=torch.full((batch_size,),opt.n_choices-1,device=device,dtype=torch.long)
    prop[:,1:]=one_hot(tool_choice_init,opt.n_choices)
    pchoice=prop[:,1:]
    message=torch.zeros(opt.max_length,batch_size,opt.vocab_size,device=device)
    # ALL starting are 0 up to the end of the sentence (no EOS)
    message[:,:,0]=1
    message.requires_grad=True
    pmessage=message
    assert pchoice.requires_grad==True
    assert pmessage.requires_grad==True
    assert prop.requires_grad==True
    assert message.requires_grad==True
    return message,pmessage,prop,pchoice

def eval(opt,agent1,agent2,reward_function1,reward_function2,
            v_fsamp3D,v_fnames3D,v_flabels3D,
            v_tsamp3D,v_tnames3D,v_tlabels3D,
            idxF,idxT1,idxT2,
            start_F_array,one_is_F_array):

    agent1.eval()
    agent2.eval()
    reward_function1.eval()
    reward_function2.eval()
    n=0
    count=0
    reward_val=0.0
    max_reward_val=0.0
    normed_reward_val=0.0
    percent_right=0.0
    ag_timeA=0
    ag_timeB=0
    totA=0
    totB=0
    tot_no_agree=0
    for episode in range(opt.n_episodes_val):
        ep_idxF=idxF[episode]
        ep_idxT1=idxT1[episode]
        ep_idxT2=idxT2[episode]
        ######### SAMPLE BATCH
        fvectors,flabels,fnames,tvectors,tlabels,tnames=\
                                get_val_batch(v_fsamp3D,v_fnames3D,v_flabels3D,
                                            v_tsamp3D,v_tnames3D,v_tlabels3D,
                                            ep_idxF,ep_idxT1,ep_idxT2)
        one_is_F=bool(one_is_F_array[episode])
        start_F=bool(start_F_array[episode])

        ######### GET MAX REWARDS
        max_r1,both_rewards1=get_max_rewards(opt,fvectors,tvectors,
                                                tlabels,reward_function1)
        max_r2,both_rewards2=get_max_rewards(opt,fvectors,tvectors,
                                                tlabels,reward_function2)

        ######### DIALOGUE
        ag_time,totals,no_agree,proposal,_,_=dialogue(opt,agent1,agent2,
                                    fvectors,tvectors,start_F,one_is_F,
                                    train=False)
        ######### REWARD
        r1,normed_r1=reward_function1(fvectors,tvectors,tlabels,
                                        proposal,None,None,max_r1)
        r2,normed_r2=reward_function2(fvectors,tvectors,tlabels,
                                        proposal,None,None,max_r2)
        if not opt.norm_r:
            percent_right1=(r1==max_r1).sum(0)
            percent_right2=(r2==max_r2).sum(0)
        elif opt.norm_r==1:
            percent_right1=(r1==1.).sum(0)
            percent_right2=(r2==1.).sum(0)
        elif opt.norm_r==2:
            percent_right1=(r1==1.).sum(0)
            percent_right2=(r2==1.).sum(0)
        reward_val+=(normed_r1*max_r1+normed_r2*max_r2).sum(0)
        normed_reward_val+=(normed_r1+normed_r2).sum(0) / 2.
        max_reward_val+=(max_r1+max_r2).sum(0)
        percent_right+=float((percent_right1+percent_right2)) / 2.
        totA+=totals[0]
        totB+=totals[1]
        ag_timeA+=ag_time[0]
        ag_timeB+=ag_time[1]
        tot_no_agree+=no_agree
        count+=r1.size(0)
    val_ag_time=np.array([ag_timeA,ag_timeB])
    val_totals=np.array([totA,totB])
    agent1.train()
    agent2.train()
    reward_function1.train()
    reward_function2.train()
    mean_reward=float(reward_val/count)
    mean_normed_reward_val=float(normed_reward_val/count)
    mean_max_reward=float(max_reward_val/count)
    percent_right=percent_right/count
    tuple_values=np.array([mean_reward,mean_normed_reward_val,
                        mean_max_reward,percent_right])
    return tuple_values,tot_no_agree,val_ag_time,val_totals

def dialogue(opt,agent1,agent2,fruits_vectors,tool_vectors,start_F,
                one_is_F,train=True):

    device=fruits_vectors.device
    batch_size=fruits_vectors.size(0)
    ######### INITIALISATION of EPISODE
    message0,pmessage0,prop0,pchoice0=initialise_output_fixed_message(opt,batch_size,device)
    all_proposalA=[]
    all_log_ppropA=[]
    all_proposalB=[]
    all_log_ppropB=[]
    all_log_psentA=[]
    all_log_psentB=[]

    done=[]
    ag_timeA=0
    ag_timeB=0
    tot_nA=0
    tot_nB=0

    if one_is_F:
        agentF=agent1
        agentT=agent2
    else:
        agentF=agent2
        agentT=agent1

    ######### RESET THE HIDDEN STATES
    agentF.reset_agent()
    agentT.reset_agent()

    ######### PROCESS INPUT
    agentF.embed_input(fruits_vectors,None)
    agentT.embed_input(None,tool_vectors)

    ######### STARTING AGENT
    if start_F==1:
        agentA=agentF
        agentB=agentT
    else:
        agentA=agentT
        agentB=agentF

    ######### INITIALIZE
    propB=prop0
    messageB=message0
    pmessageB=pmessage0
    pchoiceB=pchoice0
    all_proposalB.append(propB)
    all_log_ppropB.append(torch.log(pchoiceB))
    log_psentB=masking_message(pmessageB,torch.log(pmessageB),opt.EOS_token)
    all_log_psentB.append(log_psentB)
    all_proposalA.append(None)
    all_log_ppropA.append(None)
    all_log_psentA.append(None)

    for t in range(opt.T):
        propA,messageA,log_pchoiceA,log_psentA=agentA(messageB,train)
        all_proposalA.append(propA)
        all_log_ppropA.append(log_pchoiceA)
        all_log_psentA.append(log_psentA)
        # stop the ones that have a proposal
        n_A=count_done(propA,opt.n_choices,done)
        tot_nA+=n_A
        #number of agreement at this time step
        ag_timeA+=n_A*(t+1)
        propB,messageB,log_pchoiceB,log_psentB=agentB(messageA,train)
        all_proposalB.append(propB)
        all_log_ppropB.append(log_pchoiceB)
        all_log_psentB.append(log_psentB)
        n_B=count_done(propB,opt.n_choices,done)
        tot_nB+=n_B
        #number of agreement at this time step
        ag_timeB+=n_B*(t+1)
        if len(done)==batch_size:
            break

    no_agree=batch_size-len(done)
    times=[ag_timeA,ag_timeB]
    totals=[tot_nA,tot_nB]
    proposal,log_pprop,log_psent=masking_no_term(all_proposalA,
                            all_proposalB,all_log_ppropA,all_log_ppropB,
                            all_log_psentA,all_log_psentB,opt)
    return times,totals,no_agree,proposal,log_pprop,log_psent

def train():

    opt = parse_arguments()
    opt.n_choices=opt.n_tools+1
    # TOOLS
    # tools is of shape n_types, n_samples, n_features
    tools_samp3D=np.load('%s/tsamp_%f.npy'%(opt.data_dir,opt.variance))
    tools_names3D=np.load('%s/tsamp_names_%f.npy'%(opt.data_dir,opt.variance))
    tools_labels3D=np.load('%s/tsamp_labels_%f.npy'%(opt.data_dir,opt.variance))
    tools_samp3D=torch.as_tensor(tools_samp3D, dtype=torch.float32)
    tools_labels3D=torch.as_tensor(tools_labels3D)
    all_tools_names=np.load('%s/tools_names.npy'%opt.data_dir)

    # MAPPING MATRICES
    M=np.load('%s/M.npy'%opt.data_dir)
    M_fruit=np.load('%s/M_fruit.npy'%opt.data_dir)
    M_tool=np.load('%s/M_tool.npy'%opt.data_dir)
    M=torch.as_tensor(M, dtype=torch.float32)
    M_fruit=torch.as_tensor(M_fruit, dtype=torch.float32)
    M_tool=torch.as_tensor(M_tool, dtype=torch.float32)

    # FRUITS
    fruits_samp3D=np.load('%s/fsamp_%f.npy'%(opt.data_dir,opt.variance))
    fruits_names3D=np.load('%s/fsamp_names_%f.npy'%(opt.data_dir,opt.variance))
    fruits_labels3D=np.load('%s/fsamp_labels_%f.npy'%(opt.data_dir,opt.variance))
    fruits_samp3D=torch.as_tensor(fruits_samp3D, dtype=torch.float32)
    fruits_labels3D=torch.as_tensor(fruits_labels3D)
    all_fruits_names=np.load('%s/fruits_names.npy'%opt.data_dir)

    # VALIDATION
    v_tsamp3D=np.load('%s/val_tsamp_%f.npy'%(opt.data_dir,opt.variance))
    v_tnames3D=np.load('%s/val_tsamp_names_%f.npy'%(opt.data_dir,opt.variance))
    v_tlabels3D=np.load('%s/val_tsamp_labels_%f.npy'%(opt.data_dir,opt.variance))
    v_tsamp3D=torch.as_tensor(v_tsamp3D, dtype=torch.float32)
    v_tlabels3D=torch.as_tensor(v_tlabels3D)

    v_fsamp3D=np.load('%s/val_fsamp_%f.npy'%(opt.data_dir,opt.variance))
    v_fnames3D=np.load('%s/val_fsamp_names_%f.npy'%(opt.data_dir,opt.variance))
    v_flabels3D=np.load('%s/val_fsamp_labels_%f.npy'%(opt.data_dir,opt.variance))
    v_fsamp3D=torch.as_tensor(v_fsamp3D, dtype=torch.float32)
    v_flabels3D=torch.as_tensor(v_flabels3D)

    val_suffix='%s_%d%d_seed%d_var%f'%('val',1,1,0,opt.variance)
    idxF=pickle.load(open(opt.data_dir+"/%s_idxF"%val_suffix, "rb" ))
    idxT1=pickle.load(open(opt.data_dir+"/%s_idxT1"%val_suffix, "rb" ))
    idxT2=pickle.load(open(opt.data_dir+"/%s_idxT2"%val_suffix, "rb" ))
    one_is_F_array=pickle.load(open(opt.data_dir+"/%s_one_is_F"%val_suffix,"rb" ))
    start_F_array=pickle.load(open(opt.data_dir+"/%s_start_F"%val_suffix,"rb" ))

    if opt.symmetric==0:
        # 1 is always F
        one_is_F_array[:]=1

    if opt.start_fruit==1:
        # always start agent F
        start_F_array[:]=1
    elif opt.start_fruit==0:
        # always start agent T
        start_F_array[:]=0

    # PROPERTIES
    tools_properties=np.load('%s/tools_properties.npy'%opt.data_dir)
    tools_properties=torch.as_tensor(tools_properties, dtype=torch.float32)
    fruits_properties=np.load('%s/fruits_properties.npy'%opt.data_dir)
    fruits_properties=torch.as_tensor(fruits_properties, dtype=torch.float32)

    if opt.cuda:
        device=torch.cuda.current_device()
        fruits_samp3D=fruits_samp3D.to(device)
        fruits_labels3D=fruits_labels3D.to(device)
        v_fsamp3D=v_fsamp3D.to(device)
        v_flabels3D=v_flabels3D.to(device)
        tools_samp3D=tools_samp3D.to(device)
        tools_labels3D=tools_labels3D.to(device)
        v_tsamp3D=v_tsamp3D.to(device)
        v_tlabels3D=v_tlabels3D.to(device)
        M=M.to(device)
        M_fruit=M_fruit.to(device)
        M_tool=M_tool.to(device)
        fruits_properties=fruits_properties.to(device)
        tools_properties=tools_properties.to(device)

    # DOMAIN
    setsf=pickle.load(open(opt.data_dir+"/setsf", "rb" ))
    setst=pickle.load(open(opt.data_dir+"/setst", "rb" ))
    in_tlabels=setst[0]
    in_flabels=setsf[0]
    out_tlabels=setst[1]
    out_flabels=setsf[1]
    tran_tlabels=setst[2]
    tran_flabels=setsf[2]

    opt.tool_feat_size=tools_samp3D.shape[2]
    opt.fruit_feat_size=fruits_samp3D.shape[2]

    opt.features_size=opt.rnn_h_size
    opt.im_embedding_size=opt.input_embedding_size
    opt.tool_embedding_size=opt.input_embedding_size
    # Input + Proposal + Parsed message
    opt.parsed_input_size=opt.input_embedding_size+opt.rnn_h_size
    opt.EOS_token=opt.vocab_size - 1

    agent1,agent2=create_agents(opt)

    parameters1 = filter(lambda p: p.requires_grad, agent1.parameters())
    parameters2 = filter(lambda p: p.requires_grad, agent2.parameters())
    optimizer1 = optim.RMSprop(parameters1,opt.lr)
    optimizer2 = optim.RMSprop(parameters2,opt.lr)
    reward_function1=RewardAgentReinf(M,M_tool,M_fruit,0,
                            opt.n_tools,opt.n_choices,opt.norm_r,opt.min_r)
    reward_function2=RewardAgentReinf(M,M_tool,M_fruit,1,
                            opt.n_tools,opt.n_choices,opt.norm_r,opt.min_r)
    optimizerBSL1=optim.RMSprop(reward_function1.baseline.parameters(),opt.lr)
    optimizerBSL2=optim.RMSprop(reward_function2.baseline.parameters(),opt.lr)

    if opt.cuda:
        agent1.to(device)
        agent2.to(device)
        reward_function1.to(device)
        reward_function2.to(device)

    reward_function1.normalise_rewards(fruits_properties[in_flabels,:],
                                            tools_properties,False)
    reward_function2.normalise_rewards(fruits_properties[in_flabels,:],
                                            tools_properties,False)
    monitor_r=np.zeros((opt.n_episodes,4))
    monitor_r_val=np.zeros((opt.n_episodes,4))
    m_ag_time=np.zeros((opt.n_episodes,3,2))
    m_ag_time_val=np.zeros((opt.n_episodes,3,2))
    # ENSURE FIXED CURRICULUM for a given setting (WARNING: role dependent)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    suffsave='var%f_lr%f_bs%d_gc%f_norm%d_l%d_v%d_bm%d_s%d_h%d_sym%d_seed%d'%\
                (opt.variance,opt.lr,opt.batch_size,opt.gc,opt.norm_r,
                opt.max_length,opt.vocab_size,opt.block_mess,opt.start_fruit,
                opt.history,opt.symmetric,opt.manualSeed)
    train_set=FruitsToolsDataset(fruits_samp3D,fruits_names3D,
                fruits_labels3D,
                in_flabels,tools_samp3D,tools_names3D,tools_labels3D,
                in_tlabels,opt.n_episodes,opt.n_tools,opt.batch_size)
    sampler = torch.utils.data.SequentialSampler(train_set)
    loader_train = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,shuffle=False,sampler=sampler)
    for episode, data in enumerate(loader_train):
        if episode % 100==0:
            tuple_values,n_no_agree,val_ag_time,val_totals=eval(opt,
                            agent1,agent2,reward_function1,reward_function2,
                            v_fsamp3D,v_fnames3D,v_flabels3D,
                            v_tsamp3D,v_tnames3D,v_tlabels3D,
                            idxF,idxT1,idxT2,
                            start_F_array,one_is_F_array)
            monitor_r_val[episode,:]=tuple_values
            m_ag_time_val[episode,0,:]=val_ag_time
            m_ag_time_val[episode,1,:]=val_totals
            m_ag_time_val[episode,2,:]=n_no_agree
            av_time=val_ag_time.sum()/max([val_totals.sum(),1])
            print("VAL E %d, R %.2f, Norm. R %.4f, %d %%, Time %.2f"%(episode,tuple_values[0],tuple_values[1]*100,tuple_values[3]*100,av_time))

        fruits_vectors,flabels,_,tool_vectors,tlabels,_,_=data
        tool_vectors=tool_vectors.permute(1,0,2)
        tlabels=tlabels.permute(1,0)

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizerBSL1.zero_grad()
        optimizerBSL2.zero_grad()

        ######### GET MAX REWARDS
        max_r1,_=get_max_rewards(opt,fruits_vectors,tool_vectors,tlabels,
                                                reward_function1)
        max_r2,_=get_max_rewards(opt,fruits_vectors,tool_vectors,tlabels,
                                                reward_function2)
        ######### DIALOGUE
        if opt.start_fruit==1:
            start_fruit=bool(1)
        elif opt.start_fruit==0:
            start_fruit=bool(0)
        elif opt.start_fruit==2:
            start_fruit=bool(np.random.binomial(1,0.5,size=1))

        if opt.symmetric==0:
            # agent 1 is always F
            one_is_F=bool(1)
        else:
            one_is_F=bool(np.random.binomial(1,0.5,size=1))

        ag_time,totals,no_agree,proposal,log_pprop,log_psent=\
                                dialogue(opt,agent1,agent2,fruits_vectors,
                                tool_vectors,start_fruit,one_is_F)
        m_ag_time[episode,0,:]=np.array(ag_time)
        m_ag_time[episode,1,:]=np.array(totals)
        m_ag_time[episode,2,:]=no_agree
        ######### REWARD
        loss1,loss_bsl1,r1,normed_r1=reward_function1(fruits_vectors,
                            tool_vectors,tlabels,proposal,log_pprop,
                            log_psent,max_r1)
        loss2,loss_bsl2,r2,normed_r2=reward_function2(fruits_vectors,
                            tool_vectors,tlabels,proposal,log_pprop,
                            log_psent,max_r2)

        loss1.mean(0).backward(retain_graph=True)
        loss2.mean(0).backward()
        # retain graph is not needed as agent do not interfere with bsl
        loss_bsl1.mean(0).backward()
        loss_bsl2.mean(0).backward()

        gradClamp(agent1.parameters(),opt.gc)
        gradClamp(agent2.parameters(),opt.gc)
        gradClamp(reward_function1.baseline.parameters(),opt.gc)
        gradClamp(reward_function2.baseline.parameters(),opt.gc)
        # print(totA,totB)
        optimizer1.step()
        optimizer2.step()
        optimizerBSL1.step()
        optimizerBSL2.step()

        if not opt.norm_r:
            percent_right1=(r1==max_r1).sum(0)
            percent_right2=(r2==max_r2).sum(0)
        elif opt.norm_r==1:
            percent_right1=(r1==1.).sum(0)
            percent_right2=(r2==1.).sum(0)
        elif opt.norm_r==2:
            percent_right1=(r1==1.).sum(0)
            percent_right2=(r2==1.).sum(0)
        monitor_r[episode,0]=(normed_r1*max_r1+normed_r2*max_r2).mean(0)
        monitor_r[episode,1]=(normed_r1+normed_r2).mean(0) / 2.
        monitor_r[episode,2]=(max_r1+max_r2).mean(0)
        monitor_r[episode,3]=float((percent_right1+percent_right2))/r1.size(0)/2
        av_time=np.array(ag_time).sum()/max([np.array(totals).sum(),1])

    # FINAL VALUE
    tuple_values,n_no_agree,val_ag_time,val_totals=eval(opt,
                    agent1,agent2,reward_function1,reward_function2,
                    v_fsamp3D,v_fnames3D,v_flabels3D,
                    v_tsamp3D,v_tnames3D,v_tlabels3D,
                    idxF,idxT1,idxT2,
                    start_F_array,one_is_F_array)
    monitor_r_val[episode,:]=tuple_values
    m_ag_time_val[episode,0,:]=val_ag_time
    m_ag_time_val[episode,1,:]=val_totals
    m_ag_time_val[episode,2,:]=n_no_agree
    np.save(os.path.join(opt.outf,'./r_train_%s'%suffsave), monitor_r)
    np.save(os.path.join(opt.outf,'./r_val_%s'%suffsave),monitor_r_val)
    np.save(os.path.join(opt.outf,'./ag_time_train_%s'%suffsave),m_ag_time)
    np.save(os.path.join(opt.outf,'./ag_time_val_%s'%suffsave),m_ag_time_val)

    A_name=os.path.join(opt.outf,'agent1_'+suffsave)
    torch.save(agent1.state_dict(), A_name)
    B_name=os.path.join(opt.outf,'agent2_'+suffsave)
    torch.save(agent2.state_dict(), B_name)

if __name__ == "__main__":

    train()
