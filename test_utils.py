# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from utils import *
import matplotlib
import matplotlib.pyplot as pl
import math
from sklearn.decomposition import PCA

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def relabel(M,labels,vectors,prop,dlabels):

    device=labels.device
    n_samples=labels.size(0)
    useful_feats=M.sum(-1).ne(0)
    sub_prop=prop[:,useful_feats][dlabels,:]
    sub_vectors=vectors[:,useful_feats]
    c=sub_prop.size(0)
    a=sub_prop[:,None,:].repeat((1,n_samples,1))
    b=sub_vectors[None,:,:].repeat((c,1,1))
    mse=((a-b)**2).sum(-1)
    new_labels=mse.argmin(0)
    # map back to domain labels
    new_labels=torch.as_tensor(dlabels)[new_labels].to(device)
    return new_labels

def topographic_analysis(opt,fvectors,tvectors,
                conversA,conversB,steps_agree):

    conversations=gather_convers_onehot(conversA,conversB,steps_agree,opt,
                                    True,True)
    print("Number of success",steps_agree.shape)
    max_uttA=np.ceil(steps_agree/2)
    max_uttB=np.floor(steps_agree/2)
    idx_A_least_one=(max_uttA>0).nonzero()
    idx_B_least_one=(max_uttB>0).nonzero()
    if idx_A_least_one.numel()>0:
        idx_A_least_one=idx_A_least_one.squeeze(1)
    if idx_B_least_one.numel()>0:
        idx_B_least_one=idx_B_least_one.squeeze(1)
    print("Number with utterances from A",idx_A_least_one.shape)
    print("Number with utterances from B",idx_B_least_one.shape)
    sub_idx=np.intersect1d(idx_A_least_one,idx_B_least_one)

    if sub_idx.shape[0]>0:
        idx_2d=np.ix_(sub_idx,sub_idx)
        print("Number with utterances from both",sub_idx.shape)

    t1vectors=tvectors[0,:]
    t2vectors=tvectors[1,:]
    all_inputs=torch.cat([fvectors,t1vectors,t2vectors],dim=-1)
    tools_pair=torch.cat([t1vectors,t2vectors],dim=-1)

    input_pairsim=return_i_pairsim(all_inputs)
    fruit_pairsim=return_i_pairsim(fvectors)
    tools_pairsim=return_i_pairsim(tools_pair)
    t1_pairsim=return_i_pairsim(t1vectors)
    t2_pairsim=return_i_pairsim(t2vectors)

    conv_pairsim=return_conv_pairsim(opt,conversations,steps_agree)
    convA_pairsim=return_conv_pairsim(opt,conversA,max_uttA)
    convB_pairsim=return_conv_pairsim(opt,conversB,max_uttB)
    conv_sims=[conv_pairsim,convA_pairsim,convB_pairsim]
    input_sims=[input_pairsim,fruit_pairsim,t1_pairsim,t2_pairsim,tools_pairsim]

    spearman_array=np.zeros((3,5))
    p_array=np.zeros((3,5))
    for i,c_sim in enumerate(conv_sims):
        for j,i_sim in enumerate(input_sims):
            r,p=return_spearman(c_sim,i_sim)
            spearman_array[i,j]=-r
            p_array[i,j]=p
    n_sub=sub_idx.shape[0]
    if n_sub>0:
        spearman_array_sub=np.zeros((3,5))
        p_array_sub=np.zeros((3,5))
        for i,c_sim in enumerate(conv_sims):
            for j,i_sim in enumerate(input_sims):
                r,p=return_spearman(c_sim[idx_2d],i_sim[idx_2d])
                spearman_array_sub[i,j]=-r
                p_array_sub[i,j]=p
    return spearman_array,p_array,spearman_array_sub,p_array_sub,n_sub

def return_conv_pairsim(opt,conversations,n_turns):
    # import editdistance
    n_samples=conversations.size(2)
    T=conversations.size(0)
    device=conversations.device
    conv_dist=torch.zeros((T,n_samples,n_samples),device=device)
    n_turns_2d=n_turns[:,None].repeat(1,n_samples)
    n_turns_2d=n_turns_2d.to(device)
    max_turn=torch.max(n_turns_2d,n_turns_2d.t())
    for t in range(T):
        message_t=conversations[t,:,:,:]
        eos_bool=message_t[:,:,opt.EOS_token].t()
        mess_length=(eos_bool.cumsum(dim=1).eq(0)).sum(1)
        mess_length=mess_length.clamp(max=opt.max_length-1)
        stopped=(n_turns<=t).nonzero()
        continued=(n_turns>t).nonzero()
        np_message_t=message_t.detach().cpu().numpy()
        np_symbols_t=np_message_t.argmax(-1)
        # fill those with the last character of the alphabet,
        # and set the max_length to max_length
        if stopped.numel()>0:
            stopped=stopped.squeeze(1)
            np_symbols_t[:,stopped]=opt.vocab_size
            mess_length[stopped]=opt.max_length-1

        if continued.numel()>0:
            continued=continued.squeeze(1)
            # distance between stopped is 0
            # let's compute the distances between continued and the rest
            for n in continued:
                l_n=mess_length[n]
                mess_n=np_symbols_t[:l_n+1,n]
                # mess_n_str=[str(i) for i in mess_n]

                # TODO: consider empty versus length 4 of # distance 4, or max # length ? for the moment 4 because edit
                # distance is at most the length of the longest sentence.)
                # so distance between continued and stop is length of continued
                if stopped.numel()>0:
                    conv_dist[t,n,stopped]=l_n.type(torch.float)+1
                    conv_dist[t,stopped,n]=l_n.type(torch.float)+1
                for m in continued:
                    if m>n:
                        l_m=mess_length[m]
                        mess_m=np_symbols_t[:l_m+1,m]
                        dist_mn=(mess_m!=mess_n).sum().item()
                        # mess_m_str=[str(i) for i in mess_m]
                        # dist_mn=editdistance.eval(mess_n_str,mess_m_str)
                        conv_dist[t,n,m]=dist_mn
    # average on all T: similarity takes conversation length into account
    conv_pairdist=conv_dist.sum(0)
    # average on max T
    idx_no_message=n_turns.eq(0).nonzero()
    print("No message number",idx_no_message.shape)
    # conv_pairdist=conv_dist.sum(0)/max_turn
    if idx_no_message.numel()>0:
        idx_no_message=idx_no_message.squeeze(1)
        # if both are empty, we consider a similarity of 1
        conv_pairdist[np.ix_(idx_no_message,idx_no_message)]=1
    return conv_pairdist

def return_length(val,symbols,max_length):
    eos_bool=symbols.eq(val)
    mess_length=(eos_bool.cumsum(dim=1).eq(0)).sum(1)
    # If no EOS, then mess_length is max_length
    mess_length=mess_length.clamp(max=max_length-1)
    return mess_length

def gather_states(opt,message,no_done,agent_state,z,mask_z):
    device=agent_state.device
    for b in no_done:
        sentence=message[:,b,:].argmax(-1)[:,None].t() # None to have 2D tensor
        length=return_length(opt.EOS_token,sentence,opt.max_length)
        for t in range(length+1):
            symb=sentence.squeeze(0)[t]
            z[b,t,symb,:]=agent_state[b,:].clone()
            mask_z[b,t,symb]=1

def gather_feat(opt,vectors,message,symbols_feat):
    batch_size=message.shape[1]
    n_feat=opt.tool_feat_size
    for b in range(batch_size):
        v1=vectors[0,b,:]
        v2=vectors[1,b,:]
        delta_feat=v2
        sentence=message[:,b,:].argmax(-1)[:,None].t() # None to have 2D tensor
        length=return_length(opt.EOS_token,sentence,opt.max_length)
        for t in range(length+1):
            symb=sentence.squeeze(0)[t]
            for f in range(n_feat):
                symbols_feat[f,t,symb]+=delta_feat[f]

def follow_action(agentA,choicesA,messagesA,messageB,input_emb,bc_sA):

    K=choicesA.shape[0]
    # broadcast message B
    bc_messageB=messageB.repeat(1,K,1)
    mess_emb=agentA.listener(bc_messageB)
    zA=agentA.body(input_emb,mess_emb,bc_sA)
    _,log_pchoiceA=agentA.proposalGen(zA,train=True)
    # compute the probability of the messages from A,
    # under the probability distribution after processing that new message
    # p(z^A|m^B,s^A_prev,i^A,a^B)=p(m^A|...)p(a^A|...)
    # p(m^A|...)
    log_p_messA_mB_prime=agentA.speaker.log_proba_input_sentence(zA,messagesA)
    # p(a^A|...)
    log_pchoiceA_mB_prime=log_pchoiceA[range(K),choicesA]
    log_pzA_mB_prime=log_pchoiceA_mB_prime+log_p_messA_mB_prime
    return log_pchoiceA_mB_prime, log_p_messA_mB_prime

def positive_listening(opt,agentA,iA,hA,messageB,K,J,Atool=False):

    device=iA.device
    # broadcast inputA and sA
    bc_sA=hA.repeat(K,1)
    # broadcast message B
    bc_messageB=messageB.repeat(1,K,1)
    # sample K z^A given this message (both actions and messages)
    mess_emb=agentA.listener(bc_messageB)
    if Atool:
        bc_inputA=iA.repeat(1,K,1)
        input_emb=agentA.tool_embedder(bc_inputA)
    else:
        bc_inputA=iA.repeat(K,1)
        input_emb=agentA.im_embedder(bc_inputA)

    zA=agentA.body(input_emb,mess_emb,bc_sA)
    messagesA,log_pmA_mB=agentA.speaker(zA,train=True)
    propsA,log_pchoiceA=agentA.proposalGen(zA,train=True)
    choicesA=propsA[:,1:].argmax(-1)
    # p(a^A|m^B)
    log_paA_mB=log_pchoiceA[range(K),choicesA]
    log_pzA_mB=log_paA_mB+log_pmA_mB
    # sample multiple messages from B TODO: choose intervention distribution
    messagesB=return_bs_message(opt,J,propsA.device)
    # for each of the J message, compute the probability of each one
    # of the K action under p(.|message)
    log_pzAj=torch.zeros(K,J,device=device)
    for j in range(J):
        mB_prime=messagesB[:,j,:].unsqueeze(1)
        log_paA_mBprime,log_pmA_mBprime=follow_action(agentA,choicesA,messagesA,
                    mB_prime,input_emb,bc_sA)
        log_pzAj[:,j]=(log_paA_mBprime+log_pmA_mBprime)
    log_pzA=torch.logsumexp(log_pzAj,dim=-1)-math.log(J)
    KL=(log_pzA_mB-log_pzA).sum().item()/float(K)
    return KL

def positive_listening2(opt,agentA,iA,hA,mB,K,J,Atool=False):

    device=iA.device
    # broadcast inputA and sA
    bc_sA=hA.repeat(K,1)
    if Atool:
        bc_inputA=iA.repeat(1,K,1)
        input_emb=agentA.tool_embedder(bc_inputA)
    else:
        bc_inputA=iA.repeat(K,1)
        input_emb=agentA.im_embedder(bc_inputA)
    # sample multiple messages from B
    messagesB=return_bs_message(opt,J,device)
    KL=torch.zeros(J)
    L=J
    for j in range(J):
        mB_prime=messagesB[:,j,:].unsqueeze(1)
        bc_messageB=mB_prime.repeat(1,K,1)
        # sample K z^A given this message (both actions and messages)
        mess_emb=agentA.listener(bc_messageB)
        zA=agentA.body(input_emb,mess_emb,bc_sA)
        messagesA,log_pmA_mB=agentA.speaker(zA,train=True)
        propsA,log_pchoiceA=agentA.proposalGen(zA,train=True)
        choicesA=propsA[:,1:].argmax(-1)
        # p(a^A|m^B)
        log_paA_mB=log_pchoiceA[range(K),choicesA]
        log_pzA_mB=log_paA_mB+log_pmA_mB
        # compute the marginals
        log_pzAl=torch.zeros(K,L,device=device)
        mBs_marginals=return_bs_message(opt,L,device)
        for l in range(L):
            mB_marg=mBs_marginals[:,l,:].unsqueeze(1)
            log_paA_mBmarg,log_pmA_mBmarg=follow_action(agentA,choicesA,
                        messagesA,mB_marg,input_emb,bc_sA)
            log_pzAl[:,l]=(log_paA_mBmarg+log_pmA_mBmarg)
        log_pzA=torch.logsumexp(log_pzAl,dim=-1)-math.log(L)
        KL[l]=(log_pzA_mB-log_pzA).sum().item()/float(K)
    IC=KL.sum()/float(J)
    return IC

def count_done_agent(proposalA,n_choices,done,done_agent):
    # it's terminated if a choice is made, e.g. null is 0
    idx=(1-proposalA[:,n_choices]).nonzero()
    if idx.numel()>0:
        idx=idx.squeeze(1)
    idx=idx.tolist()
    res=list(set(idx).difference(done))
    done.extend(res)
    done_agent[res]=1
    count=len(res)
    return count,res

def gather_sent(opt,labels,message,symbols):
    batch_size=message.shape[1]
    for b in range(batch_size):
        label=labels[b]
        sentence=message[:,b,:].argmax(-1)[:,None].t() # None to have 2D tensor
        length=return_length(opt.EOS_token,sentence,opt.max_length)
        for t in range(length+1):
            symb=sentence.squeeze(0)[t]
            symbols[label,t,symb]+=1

def return_batch_fruit(opt, f,domain_flabels,domain_tlabels,
                        fruits_properties,tools_properties,
                        all_fruits_names,all_tools_names):
    device=fruits_properties.device
    n_tools=len(domain_tlabels)
    tool_feat_size=tools_properties.shape[1]
    n_comp=n_tools*(n_tools-1)
    fruit=domain_flabels[f]
    tnames1=[]
    tnames2=[]
    fruit_prop=fruits_properties[fruit,:].unsqueeze(0)
    fvectors=fruit_prop.repeat(n_comp,1)
    fnames=np.array([all_fruits_names[fruit]]*n_comp)
    flabels=torch.full((n_comp,),fruit,device=device,dtype=torch.long)
    # each batch is 1 fruit, and all possible comparisons between
    # DIFFERENT tools
    tvectors=torch.zeros((opt.n_tools,n_comp,tool_feat_size),device=device)
    tlabels=torch.zeros((opt.n_tools,n_comp),device=device,dtype=torch.long)
    b=0
    for t1 in range(n_tools):
        tool1=domain_tlabels[t1]
        tool1_prop=tools_properties[tool1,:]
        for t2 in range(n_tools):
            tool2=domain_tlabels[t2]
            if tool2!=tool1:
                tool2_prop=tools_properties[tool2,:]
                tvectors[0,b,:]=tool1_prop
                tvectors[1,b,:]=tool2_prop
                tlabels[0,b]=tool1
                tlabels[1,b]=tool2
                tnames1.append(all_tools_names[tool1])
                tnames2.append(all_tools_names[tool2])
                b+=1
    tnames1=np.array(tnames1)[np.newaxis]
    tnames2=np.array(tnames2)[np.newaxis]
    tnames=np.concatenate([tnames1,tnames2],axis=0).squeeze()
    return fvectors,flabels,fnames,tvectors,tlabels,tnames

def map_kl(kl_12mat,kl_21mat,av_n):
    device=kl_12mat.device
    array_ft=torch.cat([kl_12mat[:,1],kl_21mat[:,0]])
    array_tf=torch.cat([kl_12mat[:,0],kl_21mat[:,1]])
    array_ab=torch.as_tensor([kl_12mat[0,0],kl_21mat[0,1],
                                kl_21mat[1,0],kl_12mat[1,1]])
    array_ba=torch.as_tensor([kl_21mat[0,0],kl_12mat[0,1],
                                kl_12mat[1,0],kl_21mat[1,1]])

    kl_ft=safe_div(array_ft.sum(),av_n).to(device)
    kl_tf=safe_div(array_tf.sum(),av_n).to(device)
    kl_ab=safe_div(array_ab.sum(),av_n).to(device)
    kl_ba=safe_div(array_ba.sum(),av_n).to(device)
    kl_12=safe_div(kl_12mat.sum(),av_n).to(device)
    kl_21=safe_div(kl_21mat.sum(),av_n).to(device)
    return kl_12,kl_21,kl_ft,kl_tf,kl_ab,kl_ba

def embeddings2(opt,colors,z,mask_z,labels,names,steps_agree,agent_id):

    ulabels=torch.unique(labels)
    agreed=torch.zeros_like(steps_agree,dtype=torch.long)
    for t in range(opt.T):
        t_agreed=(steps_agree<=t*2+agent_id)
        agreed[t_agreed]=1
        cat=0
        idx=None
        z_t=None
        # gather all messages
        for l in range(opt.max_length):
            for v in range(opt.vocab_size):
                idx_lv=mask_z[t,:,l,v].nonzero()
                if idx_lv.numel()>0:
                    idx_lv=idx_lv.squeeze(1)
                    z_lv=z[t,idx_lv,l,v,:]
                    symb_lv=v*torch.ones(idx_lv.shape[0],dtype=torch.uint8)
                    assert z_lv.eq(0).nonzero().numel()==0
                    if not cat:
                        z_t=z_lv
                        idx=idx_lv
                        cat=1
                        rec_symb=symb_lv
                    else:
                        z_t=torch.cat([z_t,z_lv],dim=0)
                        idx=torch.cat([idx,idx_lv],dim=0)
                        rec_symb=torch.cat([rec_symb,symb_lv],dim=0)

        if z_t is not None and z_t.shape[0]>1:
            normed_Z=(z_t-z_t.mean(0))/z_t.std(0)
            normed_Z[normed_Z != normed_Z] = 0
            normed_Z[normed_Z.abs() == np.inf] = 0
            normed_Z=normed_Z.detach().cpu().numpy()

            labels_t=labels[idx]
            agreed_t=agreed[idx]
            Z_pca=PCA(n_components=2).fit_transform(normed_Z)
            fig, ax = pl.subplots(figsize=(8,8))
            for _,label in enumerate(ulabels):
                s='%s'%names[label]
                # indexes refers to the subset idx
                indexes=labels_t.eq(label).nonzero()
                if indexes.numel()>0:
                    indexes=indexes.squeeze(1)
                    non_stop=(1-agreed_t)[indexes].nonzero()
                    stop=agreed_t[indexes].nonzero()
                    if stop.numel()>0:
                        stop=stop.squeeze(1)
                        data_x=Z_pca[indexes[stop],0]
                        data_y=Z_pca[indexes[stop],1]
                        ax.scatter(data_x,data_y,marker='s',
                            color=colors[:,label],alpha=0.9,label=s)
                        if indexes[stop].shape[0]==1:
                            data_x=np.array([data_x])
                            data_y=np.array([data_y])
                        n=data_x.shape[0]
                        for i in range(n):
                            j=indexes[stop][i]
                            pl.text(data_x[i],data_y[i],s=str(rec_symb[j].item()))
                    if non_stop.numel()>0:
                        non_stop=non_stop.squeeze(1)
                        data_x=Z_pca[indexes[non_stop],0]
                        data_y=Z_pca[indexes[non_stop],1]
                        ax.scatter(data_x,data_y,
                            color=colors[:,label],alpha=0.9,label=s)
                        if indexes[non_stop].shape[0]==1:
                            data_x=np.array([data_x])
                            data_y=np.array([data_y])
                        n=data_x.shape[0]
                        for i in range(n):
                            j=indexes[non_stop][i]
                            pl.text(data_x[i],data_y[i],s=str(rec_symb[j].item()))
            ax.set_title("Turn %d"%t)
            ax.legend(loc='center left',bbox_to_anchor=(1, 0.5))
            pl.show()

def embeddings(opt,colors,z,mask_z,labels,names,steps_agree,agent_id):

    ulabels=torch.unique(labels)
    agreed=torch.zeros_like(steps_agree,dtype=torch.long)
    mask_z_all=mask_z
    z_all=z
    for t in range(opt.T):
        t_agreed=(steps_agree<=t*2+agent_id)
        agreed[t_agreed]=1
        for l in range(opt.max_length):
            fig, ax = pl.subplots(figsize=(8,8))
            rec_symb=np.zeros(steps_agree.size(0),dtype=np.uint8)
            for v in range(opt.vocab_size):
                idx=mask_z_all[t,:,l,v].nonzero()
                if idx.numel()>0:
                    idx=idx.squeeze(1)
                    z_lv=z_all[t,idx,l,v,:]
                    normed_Z=(z_lv-z_lv.mean(0))/z_lv.std(0)
                    normed_Z[normed_Z != normed_Z] = 0
                    normed_Z[normed_Z.abs() == np.inf] = 0
                    normed_Z=normed_Z.detach().cpu().numpy()

                    labels_lv=labels[idx]
                    agreed_lv=agreed[idx]
                    if normed_Z.shape[0]>1:
                        Z_pca=PCA(n_components=2).fit_transform(normed_Z)
                        for _,label in enumerate(ulabels):
                            s='%s'%names[label]
                            # indexes refers to the subset idx
                            indexes=labels_lv.eq(label).nonzero()
                            if indexes.numel()>0:
                                indexes=indexes.squeeze(1)
                            non_stop=(1-agreed_lv[indexes]).nonzero()
                            stop=agreed_lv[indexes].nonzero()
                            if non_stop.numel()>0:
                                non_stop=non_stop.squeeze(1)
                                data_x=Z_pca[indexes[non_stop],0]
                                data_y=Z_pca[indexes[non_stop],1]
                                ax.scatter(data_x,data_y,
                                    color=colors[:,label],alpha=0.9,label=s)
                                if indexes[non_stop].shape[0]==1:
                                    data_x=np.array([data_x])
                                    data_y=np.array([data_y])
                                n=data_x.shape[0]
                                for i in range(n):
                                    pl.text(data_x[i],data_y[i],s=str(v))
                            if stop.numel()>0:
                                stop=stop.squeeze(1)
                                data_x_stop=Z_pca[indexes[stop],0]
                                data_y_stop=Z_pca[indexes[stop],1]
                                ax.scatter(data_x_stop,data_y_stop,marker='s',
                                    color=colors[:,label],alpha=0.9,label=s)

            ax.set_title("Turn %d"%t)
            ax.legend(loc='center left',bbox_to_anchor=(1, 0.5))
            pl.show()

def pl_BA(opt,t,agentA,hA_old,messageB,no_doneB,iAvectors,Atool,no_doneA,new_A,
                    K=10,J=10):
    kl=torch.zeros(messageB.shape[1],device=messageB.device)
    for idx in no_doneB:
        mB=messageB[:,idx,:].unsqueeze(1)
        if Atool:
            iA=iAvectors[:,idx,:].unsqueeze(1)
        else:
            iA=iAvectors[idx,:].unsqueeze(0)
        if opt.history:
            if hA_old is None:
                hA=agentA.h0.to(messageB.device)
            else:
                hA=hA_old[idx,:].unsqueeze(0)
        else:
            hA=agentA.h0.to(messageB.device)
        kl[idx]=positive_listening(opt,agentA,iA,hA,mB,K,J,Atool)
    if len(no_doneA)>0:
        kl_cont=kl[no_doneA]
    else:
        kl_cont=0
    if len(new_A)>0:
        kl_stop=kl[new_A]
    else:
        kl_stop=0
    return kl,kl_cont,kl_stop

def gather_set_of_symbs(opt,symbols,labels):

    symbs=[]
    n_class=symbols.shape[1]
    used_symbols=(symbols>0).permute(1,0,2,3)
    for c in range(n_class):
        this_c_symbs=[]
        for t in range(opt.T):
            for l in range(opt.max_length):
                this_c_symbs_tl=used_symbols[c,t,l,:].nonzero()
                if this_c_symbs_tl.numel()>0:
                    this_c_symbs+=this_c_symbs_tl.squeeze(1).tolist()
        symbs.append(np.unique(this_c_symbs))
    return symbs

def gather_convers_onehot(conversA,conversB,steps_agree,opt,\
                                    use_agentA=True,use_agentB=True):
    N=steps_agree.size(0)
    device=conversA.device
    if use_agentA and use_agentB:
        top_length=opt.T*2
    else:
        top_length=opt.T

    conversations=torch.full((top_length,opt.max_length,N,opt.vocab_size),
                                    -1,device=device,dtype=torch.float)
    k=0
    for t in range(opt.T):
        if use_agentA:
            batch_idx=(steps_agree>2*t).nonzero()
            if batch_idx.numel()>0:
                batch_idx=batch_idx.squeeze(1)
                batch_idx=batch_idx.tolist()
                utt=conversA[t,:,batch_idx,:]
                conversations[k,:,batch_idx,:]=conversA[t,:,batch_idx,:]
                k+=1
        if use_agentB:
            batch_idx=(steps_agree>(2*t+1)).nonzero()
            if batch_idx.numel()>0:
                batch_idx=batch_idx.squeeze(1)
                batch_idx=batch_idx.tolist()
                conversations[k,:,batch_idx,:]=conversB[t,:,batch_idx,:]
                k+=1
    return conversations
