# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import pdb
import datetime
from architectures import *
from utils import *
from train import *
from test_utils import *
from semantical_analysis import *

def gather_symbols(opt,proposalA,proposalB,flabels,tlabels,start_F,one_is_F,\
            utterancesA,utterancesB,symbols1F,symbols1T,symbols2F,symbols2T):

    binary=1.-proposalB[0][:,opt.n_choices].unsqueeze(1)
    mask_t=(1.-binary)
    # 0th step, all proposal are from B
    proposal=mask_t*proposalB[0]
    n_turns=max([len(proposalA),len(proposalB)])
    if start_F: # start with fruit
        labelsA=flabels
        labelsB=tlabels[0,:]
        if one_is_F: # agent 1 is fruit
            symbolsA=symbols1F[0,:]
            symbolsB=symbols2T[1,:]
        else:
            symbolsA=symbols2F[0,:]
            symbolsB=symbols1T[1,:]
    else: # start with tool
        labelsA=tlabels[0,:]
        labelsB=flabels
        if one_is_F: # agent 2 is tool
            symbolsA=symbols2T[0,:]
            symbolsB=symbols1F[1,:]
        else: # agent 1 is tool
            symbolsA=symbols1T[0,:]
            symbolsB=symbols2F[1,:]
    for t in range(1,n_turns):
        ######## A
        proposal=mask_t*proposalA[t]+(1.-mask_t)*proposal
        # update termination
        binary=1.-proposalA[t][:,opt.n_choices].unsqueeze(1)
        mask_t=mask_t*(1.-binary) # those get fixed
        pmask_t=mask_t.squeeze(1)
        no_doneA=(pmask_t==1).nonzero()
        if no_doneA.numel()>0:
            no_doneA=no_doneA.squeeze(1)
            gather_sent(opt,labelsA[no_doneA],utterancesA[t-1,:,no_doneA,:],
                        symbolsA[t-1,:,:,:])
        ######## B
        proposal=mask_t*proposalB[t]+(1.-mask_t)*proposal
        binary=1.-proposalB[t][:,opt.n_choices].unsqueeze(1)
        mask_t=mask_t*(1.-binary) # those get fixed
        pmask_t=mask_t.squeeze(1)
        no_doneB=(pmask_t==1).nonzero()
        if no_doneB.numel()>0:
            no_doneB=no_doneB.squeeze(1)
            gather_sent(opt,labelsB[no_doneB],utterancesB[t-1,:,no_doneB,:],
                        symbolsB[t-1,:,:,:])
def test_dialogue(opt,agent1,agent2,fvectors,tvectors,flabels,fnames,
            tlabels,tnames,symbols1F,symbols1T,symbols2F,symbols2T,
            M_tool,start_F=True,one_is_F=True,pl=False):

    device=fvectors.device
    batch_size=fvectors.size(0)
    if opt.cuda:
        tensor_idx=torch.cuda.LongTensor(range(batch_size),
                    device=device).unsqueeze(1)
    else:
        tensor_idx=torch.LongTensor(range(batch_size)).unsqueeze(1)
    ######### INITIALISATION of EPISODE
    message0,pmessage0,prop0,pchoice0=initialise_output_fixed_message(opt,batch_size,device)
    utterancesA=torch.full((opt.T,opt.max_length,batch_size,opt.vocab_size),
                                                -1,device=device)
    utterancesB=torch.full((opt.T,opt.max_length,batch_size,opt.vocab_size),
                                                -1,device=device)

    all_proposalA=[]
    all_log_ppropA=[]
    all_proposalB=[]
    all_log_ppropB=[]
    all_log_psentA=[]
    all_log_psentB=[]
    done=[]
    doneA=torch.zeros(batch_size,dtype=torch.uint8)
    doneB=torch.zeros(batch_size,dtype=torch.uint8)
    ag_timeA=0
    ag_timeB=0
    tot_nA=0
    tot_nB=0
    steps_agree=torch.zeros(batch_size).fill_(2*opt.T)
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
    agentF.embed_input(fvectors,None)
    agentT.embed_input(None,tvectors)

    ######### STARTING AGENT
    if start_F:
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
    no_doneB=torch.arange(batch_size).tolist()
    kl_AB=torch.zeros(opt.T) # effect of A on B
    kl_BA=torch.zeros(opt.T) # effect of B on A
    kl_AB_cont=torch.zeros(opt.T)
    kl_AB_stop=torch.zeros(opt.T)
    kl_BA_cont=torch.zeros(opt.T)
    kl_BA_stop=torch.zeros(opt.T)
    batch_kl_AB=torch.zeros(opt.T,batch_size,device=device)
    batch_kl_BA=torch.zeros(opt.T,batch_size,device=device)
    count_kl=torch.zeros(2)
    count_kl_cont=torch.zeros(2)
    count_kl_stop=torch.zeros(2)
    ######### DIALOGUE
    for t in range(opt.T):
        ########## AGENT A
        hA_old=agentA.h
        propA,messageA,log_pchoiceA,log_psentA=agentA(messageB,opt.sample_test)
        all_proposalA.append(propA)
        all_log_ppropA.append(log_pchoiceA)
        all_log_psentA.append(log_psentA)
        # get the indices of accepted by A
        n_A,new_A=count_done_agent(propA,opt.n_choices,done,doneA)
        no_doneA=[b for b in range(batch_size) if b not in done]
        if pl and len(no_doneB)>0:
            if start_F:
                vectors=fvectors
                Atool=False
            else:
                vectors=tvectors
                Atool=True
            kl,kl_cont,kl_stop=pl_BA(opt,t,agentA,hA_old,messageB,no_doneB,
                                        vectors,Atool,no_doneA,new_A)
            kl_BA[t]=kl.sum(0)/len(no_doneB)
            batch_kl_BA[t,no_doneB]=kl[no_doneB]
            count_kl[0]+=1
            if len(no_doneA)>0:
                kl_BA_cont[t]=kl_cont.sum(0)/len(no_doneA)
                count_kl_cont[0]+=1
            if len(new_A)>0:
                kl_BA_stop[t]=kl_stop.sum(0)/len(new_A)
                count_kl_stop[0]+=1
        tot_nA+=n_A
        # number of agreement at this time step
        ag_timeA+=n_A*(t+1)
        steps_agree[new_A]=t*2
        # gather utterances and symbols
        if len(no_doneA)>0:
            utterancesA[t,:,no_doneA,:]=messageA[:,no_doneA,:]
        ########## AGENT B
        hB_old=agentB.h
        propB,messageB,log_pchoiceB,log_psentB=agentB(messageA,opt.sample_test)
        # get the indices of stopped by B
        n_B, new_B=count_done_agent(propB,opt.n_choices,done,doneB)
        no_doneB=[b for b in range(batch_size) if b not in done]
        all_proposalB.append(propB)
        all_log_ppropB.append(log_pchoiceB)
        all_log_psentB.append(log_psentB)
        if pl and len(no_doneA)>0:
            if start_F:
                vectors=tvectors
                Atool=True
            else:
                vectors=fvectors
                Atool=False
            kl,kl_cont,kl_stop=pl_BA(opt,t,agentB,hB_old,messageA,no_doneA,
                                        vectors,Atool,no_doneB,new_B)
            kl_AB[t]=kl.sum(0)/len(no_doneA)
            batch_kl_AB[t,no_doneA]=kl[no_doneA]
            count_kl[1]+=1
            if len(no_doneB) > 0:
                kl_AB_cont[t]=kl_cont.sum(0)/len(no_doneB)
                count_kl_cont[1]+=1
            if len(new_B) > 0:
                kl_AB_stop[t]=kl_stop.sum(0)/len(new_B)
                count_kl_stop[1]+=1
        tot_nB+=n_B
        # number of agreement at this time step
        ag_timeB+=n_B*(t+1)
        steps_agree[new_B]=t*2+1
        if len(no_doneB)>0:
            utterancesB[t,:,no_doneB,:]=messageB[:,no_doneB,:]
    # fill no agree with T*2
    steps_agree[no_doneB]=opt.T*2
    tots=[tot_nA,tot_nB]
    dones=[doneA,doneB]
    no_agree=batch_size-len(done)
    times=[ag_timeA,ag_timeB]
    proposal,log_pprop,log_psent=masking_no_term(all_proposalA,
                            all_proposalB,all_log_ppropA,all_log_ppropB,
                            all_log_psentA,all_log_psentB,opt)
    kl_AB=safe_div(kl_AB.sum(0),count_kl[1])
    kl_BA=safe_div(kl_BA.sum(0),count_kl[0])
    kl_AB_cont=safe_div(kl_AB_cont.sum(0),count_kl_cont[1])
    kl_BA_cont=safe_div(kl_BA_cont.sum(0),count_kl_cont[0])
    kl_AB_stop=safe_div(kl_AB_stop.sum(0),count_kl_stop[1])
    kl_BA_stop=safe_div(kl_BA_stop.sum(0),count_kl_stop[0])
    theta=0.1
    dial_AB=(batch_kl_AB>theta).sum(0)>0 # if any turn is > theta, will be 1
    dial_BA=(batch_kl_BA>theta).sum(0)>0 # if any turn is > theta, will be 1
    n_dialogue_act=((dial_AB+dial_BA).eq(2)).type(torch.float).mean(0)
    gather_symbols(opt,all_proposalA,all_proposalB,flabels,tlabels,start_F,
    one_is_F,utterancesA,utterancesB,symbols1F,symbols1T,symbols2F,symbols2T)
    return times,tots,dones,no_agree,proposal,utterancesA,utterancesB,\
            steps_agree,kl_AB,kl_BA,n_dialogue_act,\
            kl_AB_cont,kl_BA_cont,kl_AB_stop,kl_BA_stop

def test(opt,suffix_s,setting=[0,0],pl=False,sem=False,topo=False,test_seed=0,
        force_one_is_F=None,force_start_F=None):

    normalise=False
    type='test'
    test_suffix='%s_%d%d_seed%d_var%f'%(type,setting[0],setting[1],0,opt.variance)
    idxF=pickle.load(open(opt.data_dir+"/%s_idxF"%test_suffix, "rb" ))
    idxT1=pickle.load(open(opt.data_dir+"/%s_idxT1"%test_suffix, "rb" ))
    idxT2=pickle.load(open(opt.data_dir+"/%s_idxT2"%test_suffix, "rb" ))
    one_is_F_array=pickle.load(open(opt.data_dir+"/%s_one_is_F"%test_suffix, "rb" ))
    start_F_array=pickle.load(open(opt.data_dir+"/%s_start_F"%test_suffix,"rb" ))
    one_is_F_array=torch.as_tensor(one_is_F_array,dtype=torch.float)
    start_F_array=torch.as_tensor(start_F_array,dtype=torch.float)
    assert start_F_array.size(0)==one_is_F_array.size(0)
    max_games=start_F_array.size(0)
    print("Max games",max_games)
    setsf=pickle.load(open(opt.data_dir+"/setsf", "rb" ))
    setst=pickle.load(open(opt.data_dir+"/setst", "rb" ))
    tools_names=np.load('%s/tools_names.npy'%opt.data_dir)
    fruits_names=np.load('%s/fruits_names.npy'%opt.data_dir)
    n_all_fruits=len(fruits_names)
    n_all_tools=len(tools_names)
    # TEST
    tools_samp3D=np.load('%s/%s_tsamp_%f.npy'%(opt.data_dir,type,opt.variance))
    tools_names3D=np.load('%s/%s_tsamp_names_%f.npy'%(opt.data_dir,type,opt.variance))
    tools_labels3D=np.load('%s/%s_tsamp_labels_%f.npy'%(opt.data_dir,type,opt.variance))
    tools_samp3D=torch.as_tensor(tools_samp3D, dtype=torch.float32)
    tools_labels3D=torch.as_tensor(tools_labels3D, dtype=torch.long)

    fruits_samp3D=np.load('%s/%s_fsamp_%f.npy'%(opt.data_dir,type,opt.variance))
    fruits_names3D=np.load('%s/%s_fsamp_names_%f.npy'%(opt.data_dir,type,opt.variance))
    fruits_labels3D=np.load('%s/%s_fsamp_labels_%f.npy'%(opt.data_dir,type,opt.variance))
    fruits_samp3D=torch.as_tensor(fruits_samp3D, dtype=torch.float32)
    fruits_labels3D=torch.as_tensor(fruits_labels3D, dtype=torch.long)
    # MAPPING MATRICES
    M=np.load('%s/M.npy'%opt.data_dir)
    M_fruit=np.load('%s/M_fruit.npy'%opt.data_dir)
    M_tool=np.load('%s/M_tool.npy'%opt.data_dir)

    # PROPERTIES
    tools_properties=np.load('%s/tools_properties.npy'%opt.data_dir)
    tools_properties=torch.as_tensor(tools_properties, dtype=torch.float32)
    fruits_properties=np.load('%s/fruits_properties.npy'%opt.data_dir)
    fruits_properties=torch.as_tensor(fruits_properties, dtype=torch.float32)

    M=torch.as_tensor(M, dtype=torch.float32)
    M_fruit=torch.as_tensor(M_fruit, dtype=torch.float32)
    M_tool=torch.as_tensor(M_tool, dtype=torch.float32)
    opt.tool_feat_size=tools_samp3D.shape[2]
    opt.fruit_feat_size=fruits_samp3D.shape[2]
    opt.features_size=opt.rnn_h_size
    opt.im_embedding_size=opt.input_embedding_size
    opt.tool_embedding_size=opt.input_embedding_size
    # Input + Proposal + Parsed message
    opt.parsed_input_size=opt.input_embedding_size+opt.rnn_h_size
    opt.EOS_token=opt.vocab_size - 1

    in_flabels=setsf[0]
    domain_flabels=setsf[setting[0]]
    domain_tlabels=setst[setting[1]]
    print("N Fruits %d, N Tools %d"%(len(domain_flabels),len(domain_tlabels)))
    if opt.cuda:
        device=torch.cuda.current_device()
        fruits_samp3D=fruits_samp3D.to(device)
        fruits_labels3D=fruits_labels3D.to(device)
        tools_samp3D=tools_samp3D.to(device)
        tools_labels3D=tools_labels3D.to(device)
        M=M.to(device)
        M_fruit=M_fruit.to(device)
        M_tool=M_tool.to(device)
        fruits_properties=fruits_properties.to(device)
        tools_properties=tools_properties.to(device)
    else:
        device=tools_properties.device

    av_predictor=PredictAverageReward(M,M_tool,M_fruit,
                                fruits_properties,tools_properties,opt.min_r)
    av_predictor.compute_overall_comp(in_flabels,domain_tlabels)
    # RANDOM PREDICTOR
    rand_predictor=PredictRandom(opt.n_tools)
    print(opt.cuda)
    agent1,agent2=create_agents(opt)
    if suffix_s is not None:
        A_name=os.path.join(opt.modeldir,'agent1_'+suffix_s)
        B_name=os.path.join(opt.modeldir,'agent2_'+suffix_s)
        if opt.cuda:
            agent1.load_state_dict(torch.load(A_name))
            agent2.load_state_dict(torch.load(B_name))
        else:
            agent1.load_state_dict(torch.load(A_name,map_location='cpu'))
            agent2.load_state_dict(torch.load(B_name,map_location='cpu'))

    if opt.st:
        agent2=copy.deepcopy(agent1)

    ########## REWARD function
    reward_function1=RewardAgentReinf(M,M_tool,M_fruit,0,
                                opt.n_tools,opt.n_choices,opt.norm_r,opt.min_r)
    reward_function2=RewardAgentReinf(M,M_tool,M_fruit,1,
                                opt.n_tools,opt.n_choices,opt.norm_r,opt.min_r)
    ### NORMALISE REWARD USING IN DOMAIN FRUITS
    reward_function1.normalise_rewards(fruits_properties[in_flabels,:],
                                            tools_properties,False)
    reward_function2.normalise_rewards(fruits_properties[in_flabels,:],
                                            tools_properties,False)
    agent1.eval()
    agent2.eval()
    reward_function1.eval()
    reward_function2.eval()
    if opt.cuda:
        agent1.to(device)
        agent2.to(device)
        reward_function1.to(device)
        reward_function2.to(device)

    # FOR SEMANTIC ANALYSIS, ROLES AND POSITIONS ARE FIXED, self-talk was done with Fruit starts and one is fruit
    # IF sem==0, the roles and positions are like training
    if opt.st: # sem == 1 uses training roles / positions
        if opt.symmetric==1: # if rained with symmetric agents, fix roles
            one_is_F_array[:]=1
        if opt.start_fruit==2: # if trained with random positions, fix positions
            assert force_start_F is not None
            start_F_array[:]=force_start_F
    if sem:
        assert force_one_is_F is not None
        one_is_F_array[:]=force_one_is_F # fix Agent roles
        if opt.start_fruit==2: # if we trained with random position
            assert force_start_F is not None
            start_F_array[:]=force_start_F # fix Agent position

    if opt.symmetric==0:
        # 1 is always F
        one_is_F_array[:]=1
    if opt.start_fruit==1:
        # always start agent F
        start_F_array[:]=1
    elif opt.start_fruit==0:
        # always start agent T
        start_F_array[:]=0

    test_av_percent=0.0
    test_rand_percent=0.0
    test_reward=0.0
    normed_test_reward=0.0
    kl_12mat=torch.zeros(2,2) # effect of 1 on 2
    kl_21mat=torch.zeros(2,2) # effect of 2 on 1
    kl_12_cont=torch.zeros(2,2)
    kl_21_cont=torch.zeros(2,2)
    kl_12_stop=torch.zeros(2,2)
    kl_21_stop=torch.zeros(2,2)
    tot_ag_av=0
    tot_ag_av1=0
    tot_ag_av2=0
    tot_ag_time1=0
    tot_ag_time2=0
    n_ties=0

    symbols1F=torch.zeros((2,opt.T,n_all_fruits,
                opt.max_length,opt.vocab_size),device=device)
    symbols1T=torch.zeros((2,opt.T,n_all_tools,
                opt.max_length,opt.vocab_size),device=device)
    symbols2F=torch.zeros((2,opt.T,n_all_fruits,
                opt.max_length,opt.vocab_size),device=device)
    symbols2T=torch.zeros((2,opt.T,n_all_tools,
                opt.max_length,opt.vocab_size),device=device)
    sum_n_f=torch.zeros(n_all_fruits,dtype=torch.long,device=device)
    per_f_r=torch.zeros(n_all_fruits,device=device)
    per_f_perc=torch.zeros(n_all_fruits,device=device)

    n_fruits=len(domain_flabels)
    n_tools=len(domain_tlabels)
    count=0

    random.seed(test_seed)
    torch.manual_seed(test_seed)
    np.random.seed(test_seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(test_seed)
        cudnn.benchmark=True
    n_games=min([len(idxF),max_games])
    tmp_kl_AB=0
    tmp_kl_BA=0
    for e in range(n_games):
        ep_idxF=idxF[e]
        ep_idxT1=idxT1[e]
        ep_idxT2=idxT2[e]
        fvectors,flabels,fnames,tvectors,tlabels,tnames=\
                get_val_batch(fruits_samp3D,fruits_names3D,fruits_labels3D,
                                    tools_samp3D,tools_names3D,tools_labels3D,
                                            ep_idxF,ep_idxT1,ep_idxT2)
        ######### GET MAX REWARDS
        max_r1,both_rewards1=get_max_rewards(opt,fvectors,tvectors,
                                                tlabels,reward_function1)
        max_r2,both_rewards2=get_max_rewards(opt,fvectors,tvectors,
                                                tlabels,reward_function2)
        n_ties+=(both_rewards1[:,0]==both_rewards1[:,1]).sum().item()
        av_proposal=av_predictor.predict(tlabels)
        rand_proposal=rand_predictor.predict(tlabels.size(1),device)

        one_is_F=bool(one_is_F_array[e])
        start_F=bool(start_F_array[e])
        ######### DIALOGUE
        ag_time,tots,dones,no_agree,proposal,uttA,uttB,\
        steps_batch,kl_AB,kl_BA,n_dialogue_act,\
        kl_AB_cont,kl_BA_cont,kl_AB_stop,kl_BA_stop=test_dialogue(opt,
                agent1,agent2,fvectors,tvectors,flabels,fnames,tlabels,
                tnames,symbols1F,symbols1T,symbols2F,symbols2T,M_tool,
                start_F,one_is_F,pl)
        ######### REWARD
        r1,normed_r1=reward_function1(fvectors,tvectors,tlabels,
                                        proposal,None,None,max_r1)
        r2,normed_r2=reward_function2(fvectors,tvectors,tlabels,
                                        proposal,None,None,max_r2)
        av_r,norm_av_r=reward_function1(fvectors,tvectors,tlabels,
                                        av_proposal,None,None,max_r1+max_r2)
        rand_r,norm_rand_r=reward_function1(fvectors,tvectors,tlabels,
                                        rand_proposal,None,None,max_r1+max_r2)
        if opt.norm_r:
            percent_right1=(r1==1.).type(torch.float)
            percent_right2=(r2==1.).type(torch.float)
            av_percent=(av_r==1.).type(torch.float)
            rand_percent=(rand_r==1.).type(torch.float)
        else:
            percent_right1=(r1==max_r1).type(torch.float)
            percent_right2=(r2==max_r2).type(torch.float)
            av_percent=(av_r==(max_r1+max_r1)).type(torch.float)
            rand_percent=(rand_r==(max_r1+max_r2)).type(torch.float)

        # TODO agree with average is computed not on the choice (ties exists)
        index_agreement=av_proposal[:,1:].argmax(-1)==proposal[:,1:].argmax(-1)
        ag_av=index_agreement.sum(0)
        ag_avA=index_agreement[dones[0]].sum(0)
        ag_avB=index_agreement[dones[1]].sum(0)
        tot_ag_av+=ag_av
        # 1 starts if F starts and 1 is F, or T starts and one is T
        if (start_F == one_is_F):
            utt1=uttA
            utt2=uttB
            tot_ag_av1+=ag_avA
            tot_ag_av2+=ag_avB
            dones1=dones[0]
            ag_time1=ag_time[0]
            dones2=dones[1]
            ag_time2=ag_time[1]
        else:
            utt1=uttB
            utt2=uttA
            tot_ag_av1+=ag_avB
            tot_ag_av2+=ag_avA
            dones1=dones[1]
            ag_time1=ag_time[1]
            dones2=dones[0]
            ag_time2=ag_time[0]
        tmp_kl_AB+=kl_AB
        tmp_kl_BA+=kl_BA
        if start_F: # T is B
            if one_is_F:
                kl_12mat[1,1]+=kl_AB # 1 is F and A
                kl_21mat[1,1]+=kl_BA # 2 is T and B
                kl_12_cont[1,1]+=kl_AB_cont
                kl_21_cont[1,1]+=kl_BA_cont
                kl_12_stop[1,1]+=kl_AB_stop
                kl_21_stop[1,1]+=kl_BA_stop
            else:
                kl_12mat[1,0]+=kl_BA # 1 is T and B
                kl_21mat[1,0]+=kl_AB # 2 is F and A
                kl_12_cont[1,0]+=kl_BA_cont
                kl_21_cont[1,0]+=kl_AB_cont
                kl_12_stop[1,0]+=kl_BA_stop
                kl_21_stop[1,0]+=kl_AB_stop
        else: # T is A
            if one_is_F:
                kl_12mat[0,1]+=kl_BA # 1 is F and B
                kl_21mat[0,1]+=kl_AB # 2 is T and A
                kl_12_cont[0,1]+=kl_BA_cont
                kl_21_cont[0,1]+=kl_AB_cont
                kl_12_stop[0,1]+=kl_BA_stop
                kl_21_stop[0,1]+=kl_AB_stop
            else:
                kl_12mat[0,0]+=kl_AB # 1 is T and A
                kl_21mat[0,0]+=kl_BA # 2 is F and B
                kl_12_cont[0,0]+=kl_AB_cont
                kl_21_cont[0,0]+=kl_BA_cont
                kl_12_stop[0,0]+=kl_AB_stop
                kl_21_stop[0,0]+=kl_BA_stop

        batch_reward=(normed_r1+normed_r2)/2.
        percent_right=(percent_right1+percent_right2)/2.
        for fruit in flabels.unique():
            idx_fruit=flabels.eq(fruit).nonzero().squeeze(1)
            per_f_r[fruit]+=batch_reward[idx_fruit].sum(0)
            per_f_perc[fruit]+=percent_right[idx_fruit].sum(0)
            sum_n_f[fruit]+=idx_fruit.shape[0]

        ep_normed_test_reward=(normed_r1+normed_r2).sum(0).item()/2.
        ep_test_reward=(normed_r1*max_r1+normed_r2*max_r2).sum(0).item()
        test_reward+=ep_test_reward
        normed_test_reward+=ep_normed_test_reward
        batch_one_is_F=torch.full((fvectors.size(0),),one_is_F,dtype=torch.long)
        batch_start_F=torch.full((fvectors.size(0),),start_F,dtype=torch.long)
        if e==0:
            steps_agree=steps_batch
            a_flabels=flabels
            a_tlabels=tlabels
            all_percent=percent_right
            dialogue_act=n_dialogue_act
            all_one_is_F=batch_one_is_F
            all_start_F=batch_start_F
            conversA=uttA
            conversB=uttB
            donesA=dones[0]
            donesB=dones[1]
            a_fvectors=fvectors
            a_tvectors=tvectors
        else:
            conversA=torch.cat([conversA,uttA],dim=2)
            conversB=torch.cat([conversB,uttB],dim=2)
            steps_agree=torch.cat([steps_agree,steps_batch],dim=0)
            a_flabels=torch.cat([a_flabels,flabels],dim=0)
            a_tlabels=torch.cat([a_tlabels,tlabels],dim=1)
            all_percent=torch.cat([all_percent,percent_right],dim=0)
            all_one_is_F=torch.cat([all_one_is_F,batch_one_is_F],dim=0)
            all_start_F=torch.cat([all_start_F,batch_start_F],dim=0)
            donesA=torch.cat([donesA,dones[0]],dim=0)
            donesB=torch.cat([donesB,dones[1]],dim=0)
            a_fvectors=torch.cat([a_fvectors,fvectors],dim=0)
            a_tvectors=torch.cat([a_tvectors,tvectors],dim=1)
            dialogue_act+=n_dialogue_act

        done=[b for b in range(r1.shape[0]) if (proposal[b,1] or proposal[b,2])]
        test_av_percent+=av_percent.sum(0)
        test_rand_percent+=rand_percent.sum(0)
        count+=r1.shape[0]
    n_11=torch.dot(start_F_array,one_is_F_array)
    n_10=torch.dot(start_F_array,1-one_is_F_array)
    n_01=torch.dot(1-start_F_array,one_is_F_array)
    n_00=torch.dot(1-start_F_array,1-one_is_F_array)
    pairs_n=torch.as_tensor([[n_00,n_01],[n_10,n_11]],dtype=torch.float)
    idx_neq0=pairs_n>0
    kl_12mat[idx_neq0]/=pairs_n[idx_neq0]
    kl_21mat[idx_neq0]/=pairs_n[idx_neq0]
    kl_12_cont[idx_neq0]/=pairs_n[idx_neq0]
    kl_21_cont[idx_neq0]/=pairs_n[idx_neq0]
    kl_12_stop[idx_neq0]/=pairs_n[idx_neq0]
    kl_21_stop[idx_neq0]/=pairs_n[idx_neq0]
    av_n=float(idx_neq0.sum())
    kl_12,kl_21,kl_ft,kl_tf,kl_ab,kl_ba=map_kl(kl_12mat,kl_21mat,av_n)

    kl_12_cont,kl_21_cont,kl_ft_cont,\
            kl_tf_cont,kl_ab_cont,kl_ba_cont=map_kl(kl_12_cont,kl_21_cont,av_n)

    kl_12_stop,kl_21_stop,kl_ft_stop,\
        kl_tf_stop,kl_ab_stop,kl_ba_stop=map_kl(kl_12_stop,kl_21_stop,av_n)

    print(pairs_n)

    count=float(count)
    start1=all_one_is_F*all_start_F+(1-all_one_is_F)*(1-all_start_F)
    donesA=donesA.type(torch.long)
    donesB=donesB.type(torch.long)
    dones1=donesA*start1+donesB*(1-start1)
    dones2=donesA*(1-start1)+donesB*start1
    donesF=donesA*all_start_F+donesB*(1-all_start_F)
    donesT=donesA*(1-all_start_F)+donesB*all_start_F
    dones=dones1+dones2
    dones1=dones1.type(torch.uint8)
    dones2=dones2.type(torch.uint8)
    donesA=donesA.type(torch.uint8)
    donesB=donesB.type(torch.uint8)
    donesF=donesF.type(torch.uint8)
    donesT=donesT.type(torch.uint8)
    dones=dones.type(torch.uint8)
    tot1=float(dones1.sum())
    tot2=float(dones2.sum())
    totA=float(donesA.sum())
    totB=float(donesB.sum())
    totF=float(donesF.sum())
    totT=float(donesT.sum())
    tot=float(dones.sum())
    tot_ag_av=float(tot_ag_av)/count
    n_ep_test=n_fruits
    per_f_perc=per_f_perc/sum_n_f.float()
    per_f_r=per_f_r/sum_n_f.float()
    normed_test_reward/=count
    test_percent=all_percent.sum()/count
    test_av_percent/=count
    test_rand_percent/=count
    dialogue_act/=float(n_games)
    perf_random=(count - n_ties)/count*0.5+n_ties/count
    test_percent_done1=safe_div(all_percent[dones1].sum(),tot1)
    test_percent_done2=safe_div(all_percent[dones2].sum(),tot2)
    test_percent_doneA=safe_div(all_percent[donesA].sum(),totA)
    test_percent_doneB=safe_div(all_percent[donesB].sum(),totB)
    test_percent_doneF=safe_div(all_percent[donesF].sum(),totF)
    test_percent_doneT=safe_div(all_percent[donesT].sum(),totT)
    test_percent_done=safe_div(all_percent[dones].sum(),tot)
    klsmat=(kl_12mat,kl_21mat)
    kls=(kl_12,kl_21,kl_ft,kl_tf,kl_ab,kl_ba)
    kls_cont=(kl_12_cont,kl_21_cont,kl_ft_cont,kl_tf_cont,kl_ab_cont,kl_ba_cont)
    kls_stop=(kl_12_stop,kl_21_stop,kl_ft_stop,kl_tf_stop,kl_ab_stop,kl_ba_stop)

    persAB=(test_percent_doneA,test_percent_doneB)
    persFT=(test_percent_doneF,test_percent_doneT)
    pers12=(test_percent_done1,test_percent_done2)
    totA=float(totA)/count
    totB=float(totB)/count
    tot1=float(tot1)/count
    tot2=float(tot2)/count
    totF=float(totF)/count
    totT=float(totT)/count
    totsAB=[totA,totB]
    tots12=[tot1,tot2]
    totsFT=[totF,totT]
    tots=(totsAB,totsFT,tots12)
    symbols=(symbols1F,symbols1T,symbols2F,symbols2T)
    convers=(conversA.detach(),conversB.detach())

    if pl:
        step_0=(steps_agree==0).type(torch.long)
        stop_0=step_0.sum().item()/count
        step_0_T=((1.-all_start_F)*step_0)
        stop_0_T=step_0_T.sum().item()/count
        return test_percent,test_percent_done,persFT,persAB,pers12,tots,\
        perf_random,test_av_percent,tot_ag_av,dialogue_act,steps_agree,kls,\
            klsmat,stop_0,stop_0_T
    if sem:
        # relabel, but in the same domain
        new_flabels=relabel(M_fruit,a_flabels,a_fvectors,fruits_properties,
                            domain_flabels)
        new_t1labels=relabel(M_tool,a_tlabels[0,:],a_tvectors[0,:],
                            tools_properties,domain_tlabels)
        new_t2labels=relabel(M_tool,a_tlabels[1,:],a_tvectors[1,:],
                            tools_properties,domain_tlabels)
        new_tlabels=torch.cat([new_t1labels[None,:],new_t2labels[None,:]],dim=0)
        return all_one_is_F,convers,steps_agree,new_flabels,\
                new_tlabels,fruits_names,tools_names,all_percent

    return symbols,test_percent,fruits_names,tools_names

def test_model(opt,seeds,test_comb,best=False,pl=False,test_seeds=[0],
                    theta=0.85):

    opt.n_choices=opt.n_tools+1
    base='var%f_lr%f_bs%d_gc%f_norm%d_l%d_v%d_bm%d_s%d_h%d_sym%d'%\
        (opt.variance,opt.lr,opt.batch_size,opt.gc,
        opt.norm_r,opt.max_length,opt.vocab_size,opt.block_mess,opt.start_fruit,
        opt.history,opt.symmetric)

    if opt.block_mess and opt.start_fruit and opt.history==0:
        theta=0.84

    best_seed=0
    best_val=0
    for seed in seeds:
        suffix=base+'_seed%d'%seed
        val_reward=np.load(os.path.join(opt.modeldir,'r_val_%s.npy'%suffix))
        percent_right=val_reward[-1,3]
        if percent_right >= best_val:
            best_val=percent_right
            best_seed=seed

    if best:
        take_seeds=[best_seed]
    else:
        take_seeds=[]
        for seed in seeds:
            suffix=base+'_seed%d'%seed
            val_reward=np.load(os.path.join(opt.modeldir,'r_val_%s.npy'%suffix))
            percent_right=val_reward[-1,3]
            if percent_right >= theta:
                take_seeds.append(seed)
    print("Best seed %d value %.4f: threshold %.2f"%(best_seed,best_val,theta))
    print("%d succeeding seeds at threshold %.2f"%(len(take_seeds),theta))

    n_seeds=len(take_seeds)
    if n_seeds > 0:
        all_percents=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        per_per_agent=np.zeros((len(test_comb),n_seeds,len(test_seeds),3,2))
        n_per_agent=np.zeros((len(test_comb),n_seeds,len(test_seeds),3,2))
        rand_percents=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        av_percent=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        agree_av=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        all_kl_12=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        all_kl_21=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        all_kl_12mat=np.zeros((len(test_comb),n_seeds,len(test_seeds),2,2))
        all_kl_21mat=np.zeros((len(test_comb),n_seeds,len(test_seeds),2,2))
        all_kl_ft=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        all_kl_tf=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        all_kl_ab=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        all_kl_ba=np.zeros((len(test_comb),n_seeds,len(test_seeds)))


        dialogue_act=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        all_stop_0=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        all_stop_0_T=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        av_nturns=np.zeros((len(test_comb),n_seeds,len(test_seeds)))
        for count,seed in enumerate(take_seeds):
            suffix=base+'_seed%d'%seed
            for s,test_seed in enumerate(test_seeds):
                unique_convers=[]
                set_of_symbs=[]
                for t,setting in enumerate(test_comb):
                    if pl:
                        per,per_done,perFT,perAB,per12,tots,rand,av,ag_av,\
                        n_dialogue_act,steps_agree,KLs,KLsmat,\
                        stop_0,stop_0_T=test(opt,suffix,setting,pl=pl,sem=False,
                                                topo=False,test_seed=test_seed)
                        kl_12,kl_21,kl_ft,kl_tf,kl_ab,kl_ba=KLs
                        kl_12mat,kl_21mat=KLsmat
                        totsAB,totsFT,tots12=tots
                        all_percents[t,count,s]=per
                        per_per_agent[t,count,s,0,:]=perFT
                        per_per_agent[t,count,s,1,:]=perAB
                        per_per_agent[t,count,s,2,:]=per12
                        n_per_agent[t,count,s,0,:]=totsFT
                        n_per_agent[t,count,s,1,:]=totsAB
                        n_per_agent[t,count,s,2,:]=tots12
                        rand_percents[t,count,s]=rand

                        all_kl_12[t,count,s]=kl_12
                        all_kl_21[t,count,s]=kl_21
                        all_kl_12mat[t,count,s,:]=kl_12mat
                        all_kl_21mat[t,count,s,:]=kl_21mat
                        all_kl_ft[t,count,s]=kl_ft
                        all_kl_tf[t,count,s]=kl_tf
                        all_kl_ab[t,count,s]=kl_ab
                        all_kl_ba[t,count,s]=kl_ba

                        av_percent[t,count,s]=av
                        agree_av[t,count,s]=ag_av
                        dialogue_act[t,count,s]=n_dialogue_act
                        all_stop_0[t,count,s]=stop_0
                        all_stop_0_T[t,count,s]=stop_0_T
                        av_nturns[t,count,s]=steps_agree.mean()
                    else:
                        _,per,_,_=test(opt,suffix,setting,pl=pl,sem=False,
                        topo=False,test_seed=test_seed)
                        all_percents[t,count,s]=per
    all_KLs=(all_kl_12,all_kl_21,all_kl_ft,all_kl_tf,all_kl_ab,all_kl_ba)
    all_KLsmat=(all_kl_12mat,all_kl_21mat)
    return all_percents,per_per_agent,av_percent,rand_percents,n_per_agent,\
        agree_av,all_KLs,all_KLsmat,\
        dialogue_act,all_stop_0,all_stop_0_T,av_nturns,base

def semantics_expe(seeds,resultdir=None,best=False,theta=0.85,start_F=True):

    opt=parse_arguments()
    opt.n_choices=opt.n_tools+1
    if resultdir is None:
        resultdir=opt.outf
    base='var%f_lr%f_bs%d_gc%f_norm%d_l%d_v%d_bm%d_s%d_h%d_sym%d'%\
        (opt.variance,opt.lr,opt.batch_size,opt.gc,
        opt.norm_r,opt.max_length,opt.vocab_size,opt.block_mess,
        opt.start_fruit,opt.history,opt.symmetric)
    best_seed=0
    best_val=0
    for seed in seeds:
        suffix=base+'_seed%d'%seed
        val_reward=np.load(os.path.join(opt.modeldir,'r_val_%s.npy'%suffix))
        percent_right=val_reward[-1,3]
        if percent_right >= best_val:
            best_val=percent_right
            best_seed=seed
    if best:
        take_seeds=[best_seed]
    else:
        take_seeds=[]
        for seed in seeds:
            suffix=base+'_seed%d'%seed
            val_reward=np.load(os.path.join(opt.modeldir,'r_val_%s.npy'%suffix))
            percent_right=val_reward[-1,3]
            if percent_right >= theta:
                take_seeds.append(seed)
    print("Best seed %d value %.4f: threshold %.2f"%(best_seed,best_val,theta))
    print("%d succeeding seeds at threshold %.2f"%(len(take_seeds),theta))
    dataset_seed=opt.manualSeed
    if len(take_seeds) > 0:
        accuracies=np.zeros((len(take_seeds),11,11))
        accuraciesAA=np.zeros((len(take_seeds),11,11))
        accuraciesBB=np.zeros((len(take_seeds),11,11))
        stats_proba_perf=np.zeros((len(take_seeds),4))
        for count,seed in enumerate(take_seeds):
            print(seed)
            suffix=base+'_seed%d'%seed
            all_one_is_F,convers,steps_agree,flabels,\
            tlabels,fruits_names,tools_names,allp=test(opt,suffix,setting=[0,0],pl=False,sem=True,topo=False,test_seed=0,force_one_is_F=1,force_start_F=start_F)
            conversA,conversB=convers
            acc,accAA,accBB,proba_perf=sem_analysis(opt,fruits_names,
                        tools_names,
                        allp,all_one_is_F,conversA,conversB,steps_agree,
                        flabels,tlabels,dataset_seed)

            accuracies[count,:]=acc
            accuraciesAA[count,:]=accAA
            accuraciesBB[count,:]=accBB
            stats_proba_perf[count,:]=proba_perf
    base+='dataset_seed%d'%opt.manualSeed
    print(accuraciesAA)
    np.save('%s/accuracies_%s_%d'%(resultdir,base,start_F),accuracies)
    np.save('%s/accuraciesAA_%s_%d'%(resultdir,base,start_F),accuraciesAA)
    np.save('%s/accuraciesBB_%s_%d'%(resultdir,base,start_F),accuraciesBB)
    np.save('%s/stats_proba_perf_%s_%d'%(resultdir,base,start_F),stats_proba_perf)

def test_performances(seeds,theta=0.85,resultdir=None):

    opt=parse_arguments()
    if resultdir is None:
        resultdir=opt.outf
    test_comb=[[0,0],[1,1],[2,2]]
    test_seeds=[opt.manualSeed]
    if opt.st:
        pl=False
    else:
        pl=True
    all_perfs,per_per_agent,av_perfs,rand_perfs,n_per_agent,agree_av,\
        KLs,KLsmat,dial_act,stop_0,stop_0_T,\
        av_n_turns,base=test_model(opt,seeds,test_comb,
                        best=False,pl=pl,test_seeds=test_seeds,theta=theta)

    base+='test_seed%d'%opt.manualSeed
    if opt.st:
        base+='self_talk'
    np.save('%s/mean_all_perfs_%s'%(resultdir,base),all_perfs)
    np.save('%s/av_perfs_%s'%(resultdir,base),av_perfs)
    np.save('%s/per_per_agent_%s'%(resultdir,base),per_per_agent)
    np.save('%s/n_per_agent_%s'%(resultdir,base),n_per_agent)
    np.save('%s/agree_av_%s'%(resultdir,base),agree_av)
    np.save('%s/KL_12_%s'%(resultdir,base),KLs[0])
    np.save('%s/KL_21_%s'%(resultdir,base),KLs[1])
    np.save('%s/KL_ft_%s'%(resultdir,base),KLs[2])
    np.save('%s/KL_tf_%s'%(resultdir,base),KLs[3])
    np.save('%s/KL_ab_%s'%(resultdir,base),KLs[4])
    np.save('%s/KL_ba_%s'%(resultdir,base),KLs[5])
    np.save('%s/KL_12mat_%s'%(resultdir,base),KLsmat[0])
    np.save('%s/KL_21mat_%s'%(resultdir,base),KLsmat[1])

    np.save('%s/dial_act_%s'%(resultdir,base),dial_act)
    np.save('%s/stop_0_%s'%(resultdir,base),stop_0)
    np.save('%s/stop_0_T_%s'%(resultdir,base),stop_0_T)
    np.save('%s/av_n_turns_%s'%(resultdir,base),av_n_turns)

if __name__ == "__main__":
    theta=0.50
    test_performances(range(1), theta=0.5)
