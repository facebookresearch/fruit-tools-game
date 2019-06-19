# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from utils import *
from test_utils import *
from torch.utils.data import BatchSampler, SequentialSampler
import itertools
import torch.optim as optim
import copy

def shuffle_convers(convers,steps_agree,agent_id):
    device=convers.device
    swapped_convers=torch.zeros_like(convers,device=device).fill_(-1)
    if agent_id==0: # agentA
        max_utt=np.ceil(steps_agree/2)
    else:
        max_utt=np.floor(steps_agree/2)
    batch_size=max_utt.size(0)
    max_T=convers.shape[0]
    idx=np.zeros((batch_size,max_T))
    for b in range(batch_size):
        k=int(max_utt[b])
        if k<=1: # nothing to shuffle
            swapped_convers[:,:,b,:]=convers[:,:,b,:]
            idx[b,:]=np.arange(max_T)
        else:
            # after agreement (should be all -1)
            swapped_convers[k:,:,b,:]=convers[k:,:,b,:]
            idx[b,k:]=np.arange(k,max_T)
            if k==2:
                x=[1,0]
            if k>2:
                x=np.arange(k)
                y=x.copy()
                np.random.shuffle(x) #TODO sure that numpy shuffles every time ?
            shuffle_sample=bool(np.random.binomial(1,0.5,size=1))
            if shuffle_sample: # shuffle up to agreement
                swapped_convers[:k,:,b,:]=convers[x,:,b,:]
                idx[b,:k]=x
            else: # copy same order
                swapped_convers[:k,:,b,:]=convers[:k,:,b,:]
                idx[b,:k]=np.arange(0,k)
    return swapped_convers

class Semantic_classifier(nn.Module):
    def __init__(self,vocab_size,embedding_size,rnn_h_size,data_size,max_length,
                EOS_token,n_f,n_t,type_pred):
        super(Semantic_classifier,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.rnn_h_size=rnn_h_size
        self.max_length=max_length
        self.data_size=data_size
        self.embeddingTable=nn.Linear(self.vocab_size,self.embedding_size,bias=True)
        self.EOS_token=EOS_token
        self.rnn=nn.RNN(self.embedding_size,self.rnn_h_size,1)
        self.lin_f=nn.Linear(self.data_size,n_f,bias=True)
        self.lin_t1=nn.Linear(self.data_size,n_t,bias=True)
        self.lin_t2=nn.Linear(self.data_size,n_t,bias=True)
        self.lin_ft1t2=nn.Linear(n_f+n_t+n_t,n_f*n_t*n_t,bias=True)
        self.lin_ft=nn.Linear(n_f+n_t,n_f*n_t,bias=True)
        self.lin_t=nn.Linear(n_t+n_t,n_t*n_t,bias=True)
        self.type_pred=type_pred

    def process_sentence(self,received_message,seq_embeddings,h0):
        batch_size=received_message.size(1)
        # rnn input is of size seq_length x batch_size x embedding_size
        output,hidden=self.rnn(seq_embeddings,h0)
        # here we have to stop at the first EOS, or BEFORE the first -1
        eos_bool=received_message[:,:,self.EOS_token].t()
        mess_length=(eos_bool.cumsum(dim=1).eq(0)).sum(1)
        # If no EOS, then mess_length is max_length
        mess_length=mess_length.clamp(max=self.max_length-1)
        # Take the hidden layer of EOS token or end if no EOS
        h_t=output[mess_length,range(batch_size),:]
        return h_t

    def embed(self,received_message,h0):
        # received_message is of size seq_length x batch_size x vocab_size
        batch_size=received_message.size(1)
        seq_length=received_message.size(0)

        # seq_embeddings is of size batch_size x seq_length x embedding_size
        mess_t=received_message.permute(1,0,2).contiguous().view(batch_size*seq_length,-1)
        seq_embeddingsV=self.embeddingTable(mess_t).view(batch_size,
                                                        seq_length,-1)
        # start hidden state 1 x batch_size x rnn_h_size
        repr_message=self.process_sentence(received_message,
                                seq_embeddingsV.permute(1,0,2),h0)
        return repr_message

    def forward(self,x):
        out_f=self.lin_f(x)
        out_t1=self.lin_t1(x)
        out_t2=self.lin_t2(x)
        n_t=out_t1.shape[1]
        n_f=out_f.shape[1]
        if self.type_pred in [1,2]:
            input=torch.cat([out_f,out_t1,out_t2],dim=1)
            out=self.lin_ft1t2(input)
            return out
        elif self.type_pred in [3,4,5]:
            input=torch.cat([out_f,out_t1],dim=1)
            out=self.lin_ft(input)
            return out
        elif self.type_pred==6:
            return out_f
        elif self.type_pred in [7,8]:
            input=torch.cat([out_t1,out_t2],dim=1)
            out=self.lin_t(input)
            return out
        elif self.type_pred in [9,10,11]:
            return out_t1


def sem_analysis(opt,fruits_names,tools_names,allp,all_one_is_F,
            conversA,conversB,steps_agree,all_flabels,all_tlabels,
            dataset_seed,train_seed=0):

    success=(allp==1).nonzero()
    if success.numel()>0:
        success=success.squeeze(1)
    take=success
    n_f=len(fruits_names)
    n_t=len(tools_names)
    d_flabels=all_flabels
    d_tlabels=all_tlabels
    if take.shape[0]>0:
        steps=steps_agree[take]
        f_labels=d_flabels[take]
        t_labels=d_tlabels[:,take]
        t1_labels=t_labels[0,:]
        t2_labels=t_labels[1,:]
        train_idx,val_idx,test_idx,proba_perf=create_classif_dataset_simple(opt,
                f_labels,t_labels,t1_labels,t2_labels,n_f,n_t,
                dataset_seed=dataset_seed)
        accuracies=-1*np.ones((11,11))
        accuraciesAA=-1*np.ones((11,11))
        accuraciesBB=-1*np.ones((11,11))
        for t,type_pred in enumerate(range(1,12)):
            if type_pred in [1,6,9,10]:
                acc,accAA,accBB,_,_=semantical_prediction(opt,
                        conversA[:,:,take,:],conversB[:,:,take,:],steps,
                        f_labels,t1_labels,t2_labels,train_idx,val_idx,test_idx,
                        n_f,n_t,type_pred,train_seed=train_seed)
                accuracies[t,:]=acc
                accuraciesAA[t,:]=accAA
                accuraciesBB[t,:]=accBB
    return accuracies,accuraciesAA,accuraciesBB,proba_perf

def create_classif_dataset_simple(opt,f_labels,t_labels,
            t1_labels,t2_labels,n_f,n_t,dataset_seed=0):
    device=f_labels.device
    random.seed(dataset_seed)
    torch.manual_seed(dataset_seed)
    np.random.seed(dataset_seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(dataset_seed)
        cudnn.benchmark=True
    # dataset size
    N=f_labels.shape[0]
    f_labels = f_labels.cpu()
    t_labels = t_labels.cpu()
    t1_labels = t1_labels.cpu()
    t2_labels = t2_labels.cpu()

    fruits=np.unique(f_labels)
    tools=np.unique(t_labels)
    triplabels=np.concatenate([f_labels[None,:],t_labels],axis=0)
    for f,fruit in enumerate(fruits):
        idx_f=np.where(triplabels[0,:]==fruit)[0]
        if idx_f.shape[0]>1:
            np.random.shuffle(idx_f)
            train_f=idx_f[:int(idx_f.shape[0]/2.)]
        else:
            train_f=idx_f
        if f==0:
            idx_train=train_f
        else:
            idx_train=np.concatenate([idx_train,train_f])
        idx_train=np.unique(idx_train)

    for tool in tools:
        # gather in priority the ones that are already there
        idx_there1=idx_train[np.where(triplabels[1,idx_train]==tool)[0]]
        idx_there2=idx_train[np.where(triplabels[2,idx_train]==tool)[0]]
        idx_there=np.concatenate([idx_there1,idx_there2])
        idx_there=np.unique(idx_there)
        # gather the total number of this tool
        idx_t1=np.where(triplabels[1,:]==tool)[0]
        idx_t2=np.where(triplabels[2,:]==tool)[0]
        idx_t=np.concatenate([idx_t1,idx_t2])
        idx_t=np.unique(idx_t)
        assert len(np.setdiff1d(idx_there,idx_t))==0
        tot_t=idx_t.shape[0]
        n_there=idx_there.shape[0]
        if tot_t==1: #there is only one, put in train if not there
            if n_there==0:
                idx_train=np.concatenate([idx_train,idx_t])
        else:
            if n_there<int(tot_t/2.): # if we have less than half
                # the ones that are in idx_t and not in idx_there
                missing_t_idx=np.setdiff1d(idx_t,idx_there)
                to_take=int(tot_t/2.)-n_there
                np.random.shuffle(missing_t_idx)
                train_t=missing_t_idx[:to_take]
                idx_train=np.concatenate([idx_train,train_t])
        idx_train=np.unique(idx_train)

    train_idx=np.unique(idx_train)
    valtest_idx=np.setdiff1d(range(N),train_idx)
    train_fruits=np.unique(f_labels[train_idx])
    train_tools=np.unique(t_labels[:,train_idx])
    assert len(np.setdiff1d(fruits,train_fruits))==0
    assert len(np.setdiff1d(tools,train_tools))==0

    Nval_test=valtest_idx.shape[0]
    val_idx=valtest_idx[:int(Nval_test/2)]
    test_idx=np.setdiff1d(valtest_idx,val_idx)

    val_fruits=np.unique(f_labels[val_idx])
    val_tools=np.unique(t_labels[:,val_idx])
    print("VAL Missing Fruits",np.setdiff1d(fruits,val_fruits))
    print("VAL Missing Tools",np.setdiff1d(tools,val_tools))

    test_fruits=np.unique(f_labels[test_idx])
    test_tools=np.unique(t_labels[:,test_idx])
    print("TEST Missing Fruits",np.setdiff1d(fruits,test_fruits))
    print("TEST Missing Tools",np.setdiff1d(tools,test_tools))
    assert np.intersect1d(train_idx,val_idx).shape[0]==0
    assert np.intersect1d(val_idx,test_idx).shape[0]==0
    assert np.intersect1d(train_idx,test_idx).shape[0]==0

    f_proba_perf=test_proba_perf(f_labels[train_idx],f_labels[test_idx],n_f)
    t1_proba_perf=test_proba_perf(t1_labels[train_idx],t1_labels[test_idx],n_t)
    t2_proba_perf=test_proba_perf(t2_labels[train_idx],t2_labels[test_idx],n_t)
    triplet_labels=f_labels*n_t*n_t+t1_labels*n_t+t2_labels
    ft1t1t2_proba_perf=test_proba_perf(triplet_labels[train_idx],triplet_labels[test_idx],n_f*n_t*n_t)
    proba_perf=[ft1t1t2_proba_perf,f_proba_perf,t1_proba_perf,t2_proba_perf]
    return train_idx,val_idx,test_idx,proba_perf

def semantical_prediction(opt,conversA,conversB,steps_agree,f_labels,t1_labels,
            t2_labels,train_idx,val_idx,test_idx,n_f,n_t,type_pred,
            train_seed=0,train=True):

    # predict can be a value out of 12 possibilities:
    # 1 (F,[T1,T2]): perfect triplet, with tools correct positions
    # 2 (F,T1,T2): perfect triplet, potentially with wrong positions
    # 3 (F,[T1,*]): fruit and T1 in correct position
    # 4 (F,[*,T2]): fruit and T2 in correct position
    # 5 (F,T): fruit and either tool
    # 6 (F) fruit only
    # 7 ([T1,T2]): tool pair, with tools correct position
    # 8 (T1, T2): tool pair, potentially with wrong positions
    # 9 ([T1, *]): tool 1, in correct position
    # 10 ([*, T2]): tool 2, in correct position
    # 11 (T): either one of the two tools

    device=conversA.device
    lookup=list(itertools.product(range(n_f),range(n_t),range(n_t)))
    lookup=torch.as_tensor(lookup,dtype=torch.long,device=device)
    lookup_ftool=list(itertools.product(range(n_f),range(n_t)))
    lookup_ftool=torch.as_tensor(lookup_ftool,dtype=torch.long,device=device)
    lookup_tools=list(itertools.product(range(n_t),range(n_t)))
    lookup_tools=torch.as_tensor(lookup_tools,dtype=torch.long,device=device)

    # Both
    valacc,classif=train_classifier(opt,train_idx,val_idx,conversA,conversB,
                steps_agree,n_f,n_t,type_pred,f_labels,t1_labels,t2_labels,
                lookup,lookup_ftool,lookup_tools,True,True,train_seed,train)
    # A only
    # TODO : NOT ADAPTED TO A IS T and vice-versa, e.g. always same starting agent
    valaccA,classifA=train_classifier(opt,train_idx,val_idx,conversA,conversB,
                steps_agree,n_f,n_t,type_pred,f_labels,t1_labels,t2_labels,
                lookup,lookup_ftool,lookup_tools,True,False,train_seed,train)
    # B only
    valaccB,classifB=train_classifier(opt,train_idx,val_idx,conversA,conversB,
                steps_agree,n_f,n_t,type_pred,f_labels,t1_labels,t2_labels,
                lookup,lookup_ftool,lookup_tools,False,True,train_seed,train)

    acc=test_classifier(classif,opt,steps_agree,conversA,conversB,
        test_idx,True,True,lookup,lookup_ftool,lookup_tools,
        f_labels,t1_labels,t2_labels)
    accAA=test_classifier(classifA,opt,steps_agree,conversA,conversB,
        test_idx,True,False,lookup,lookup_ftool,lookup_tools,
        f_labels,t1_labels,t2_labels)
    accBB=test_classifier(classifB,opt,steps_agree,conversA,conversB,
        test_idx,False,True,lookup,lookup_ftool,lookup_tools,
        f_labels,t1_labels,t2_labels)
    accAB=test_classifier(classifA,opt,steps_agree,conversA,conversB,
        test_idx,False,True,lookup,lookup_ftool,lookup_tools,
        f_labels,t1_labels,t2_labels)
    accBA=test_classifier(classifB,opt,steps_agree,conversA,conversB,
        test_idx,True,False,lookup,lookup_ftool,lookup_tools,
        f_labels,t1_labels,t2_labels)

    return acc,accAA,accBB,accAB,accBA

def sem_common_language(opt,conversA,conversB,one_is_F,
            steps_agree,f_labels,t1_labels,t2_labels,
            train_idx1,val_idx1,test_idx1,
            train_idx2,val_idx2,test_idx2,
            n_f,n_t,type_pred,
            train_seed=0,train=True):

    device=conversA.device
    lookup=list(itertools.product(range(n_f),range(n_t),range(n_t)))
    lookup=torch.as_tensor(lookup,dtype=torch.long,device=device)
    lookup_ftool=list(itertools.product(range(n_f),range(n_t)))
    lookup_ftool=torch.as_tensor(lookup_ftool,dtype=torch.long,device=device)
    lookup_tools=list(itertools.product(range(n_t),range(n_t)))
    lookup_tools=torch.as_tensor(lookup_tools,dtype=torch.long,device=device)

    valacc1,classif1=train_classifier(opt,train_idx1,val_idx1,
        conversA[:,:,one_is_F.eq(1),:],conversB[:,:,one_is_F.eq(1),:],
        steps_agree[one_is_F.eq(1)],n_f,n_t,type_pred,
        f_labels[one_is_F.eq(1)],
        t1_labels[one_is_F.eq(1)],t2_labels[one_is_F.eq(1)],
        lookup,lookup_ftool,lookup_tools,True,True,train_seed,train)
    valacc2,classif2=train_classifier(opt,train_idx2,val_idx2,
        conversA[:,:,one_is_F.eq(0),:],conversB[:,:,one_is_F.eq(0),:],
        steps_agree[one_is_F.eq(0)],n_f,n_t,type_pred,
        f_labels[one_is_F.eq(0)],
        t1_labels[one_is_F.eq(0)],t2_labels[one_is_F.eq(0)],
        lookup,lookup_ftool,lookup_tools,True,True,train_seed,train)
    acc11=test_classifier(classif1,opt,steps_agree[one_is_F.eq(1)],
        conversA[:,:,one_is_F.eq(1),:],conversB[:,:,one_is_F.eq(1),:],
        test_idx1,True,True,lookup,lookup_ftool,lookup_tools,
        f_labels[one_is_F.eq(1)],
        t1_labels[one_is_F.eq(1)],t2_labels[one_is_F.eq(1)])
    acc22=test_classifier(classif2,opt,steps_agree[one_is_F.eq(0)],
        conversA[:,:,one_is_F.eq(0),:],conversB[:,:,one_is_F.eq(0),:],
        test_idx2,True,True,lookup,lookup_ftool,lookup_tools,
        f_labels[one_is_F.eq(0)],
        t1_labels[one_is_F.eq(0)],t2_labels[one_is_F.eq(0)])
    acc12=test_classifier(classif1,opt,steps_agree[one_is_F.eq(0)],
        conversA[:,:,one_is_F.eq(0),:],conversB[:,:,one_is_F.eq(0),:],
        test_idx2,True,True,lookup,lookup_ftool,lookup_tools,
        f_labels[one_is_F.eq(0)],
        t1_labels[one_is_F.eq(0)],t2_labels[one_is_F.eq(0)])
    acc21=test_classifier(classif2,opt,steps_agree[one_is_F.eq(1)],
        conversA[:,:,one_is_F.eq(1),:],conversB[:,:,one_is_F.eq(1),:],
        test_idx1,True,True,lookup,lookup_ftool,lookup_tools,
        f_labels[one_is_F.eq(1)],
        t1_labels[one_is_F.eq(1)],t2_labels[one_is_F.eq(1)])
    return acc11,acc22,acc12,acc21

def test_classifier(classifier,opt,steps_agree,conversA,conversB,idx,
                    use_agentA,use_agentB,lookup,lookup_ftool,lookup_tools,
                    f_labels,t1_labels,t2_labels):

    flabels=f_labels[idx]
    t1labels=t1_labels[idx]
    t2labels=t2_labels[idx]

    N=float(idx.shape[0])
    data=embed_convers(classifier,conversA[:,:,idx,:],
                conversB[:,:,idx,:],steps_agree[idx],opt,
                use_agentA,use_agentB)
    out=classifier(data)
    _,pred=torch.max(out,1)

    accuracies=-1*np.ones(11)
    correct_f=np.array([])
    correct_t1=np.array([])
    correct_t1_either=np.array([])
    correct_t2=np.array([])
    correct_t2_either=np.array([])
    correct_t=np.array([])
    pred_f=None
    pred_t1=None
    pred_t2=None
    pred_t=None
    if classifier.type_pred in [1,2]:
        pred_f=lookup[pred,0]
        pred_t1=lookup[pred,1]
        pred_t2=lookup[pred,2]
    elif classifier.type_pred==3:
        pred_f=lookup_ftool[pred,0]
        pred_t1=lookup_ftool[pred,1]
    elif classifier.type_pred==4:
        pred_f=lookup_ftool[pred,0]
        pred_t2=lookup_ftool[pred,1]
    elif classifier.type_pred==5:
        pred_f=lookup_ftool[pred,0]
        pred_t=lookup_ftool[pred,1]
    elif classifier.type_pred==6:
        pred_f=pred
    elif classifier.type_pred in [7,8]:
        pred_t1=lookup_tools[pred,0]
        pred_t2=lookup_tools[pred,1]
    elif classifier.type_pred==9:
        pred_t1=pred
    elif classifier.type_pred==10:
        pred_t2=pred
    elif classifier.type_pred==11:
        pred_t=pred
    if pred_f is not None:
        correct_f=(pred_f==flabels).nonzero()
    if pred_t1 is not None:
        correct_t1=(pred_t1==t1labels).nonzero()
        correct_t1_either=((pred_t1==t1labels)+(pred_t1==t2labels)).nonzero()
    if pred_t2 is not None:
        correct_t2=(pred_t2==t2labels).nonzero()
        correct_t2_either=((pred_t2==t1labels)+(pred_t2==t2labels)).nonzero()
    if pred_t is not None:
        correct_t=((pred_t==t1labels)+(pred_t==t2labels)).nonzero()

    if (pred_t1 is not None) and (pred_t2 is not None):
        pair_t1t2=np.intersect1d(correct_t1,correct_t2)
        pair_t1t2_either=np.intersect1d(correct_t1_either,correct_t2_either)

    if (pred_f is not None) and (pred_t1 is not None) and (pred_t2 is not None):
        triplet_correct=np.intersect1d(correct_f,pair_t1t2)
        tuple_ft1t2_either=np.intersect1d(correct_f,pair_t1t2_either)
        accuracies[0]=triplet_correct.shape[0]/N*100.
        accuracies[1]=tuple_ft1t2_either.shape[0]/N*100.

    if (pred_f is not None) and (pred_t1 is not None):
        tuple_ft1_correct=np.intersect1d(correct_f,correct_t1)
        accuracies[2]=tuple_ft1_correct.shape[0]/N*100.

    if (pred_f is not None) and (pred_t2 is not None):
        tuple_ft2_correct=np.intersect1d(correct_f,correct_t2)
        accuracies[3]=tuple_ft2_correct.shape[0]/N*100.

    if (pred_f is not None) and (pred_t is not None):
        tuple_ft_correct=np.intersect1d(correct_f,correct_t)
        accuracies[4]=tuple_ft_correct.shape[0]/N*100.
    if (pred_f is not None):
        accuracies[5]=correct_f.shape[0]/N*100.

    if (pred_t1 is not None) and (pred_t2 is not None):
        accuracies[6]=pair_t1t2.shape[0]/N*100.
        accuracies[7]=pair_t1t2_either.shape[0]/N*100.

    if (pred_t1 is not None):
        accuracies[8]=correct_t1.shape[0]/N*100.

    if (pred_t2 is not None):
        accuracies[9]=correct_t2.shape[0]/N*100.

    if (pred_t is not None):
        accuracies[10]=correct_t.shape[0]/N*100.
    return accuracies

def train_classifier(opt,train_idx,val_idx,conversA,conversB,steps_agree,
            n_f,n_t,type_pred,f_labels,t1_labels,t2_labels,
            lookup,lookup_ftool,lookup_tools,
            use_agentA,use_agentB,train_seed,train=True):

    Ntrain=float(train_idx.shape[0])
    Nval=float(val_idx.shape[0])
    print("Ntrain",Ntrain)
    print("Nval",Nval)

    random.seed(train_seed)
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(train_seed)
        cudnn.benchmark=True

    device=lookup.device
    data_size=opt.rnn_h_size
    classifier=Semantic_classifier(opt.vocab_size,opt.symb_embedding_size,
                    opt.rnn_h_size,data_size,opt.max_length,opt.EOS_token,
                    n_f,n_t,type_pred)

    classifier=classifier.to(device)

    sampler = torch.utils.data.SequentialSampler(train_idx)
    loader_train = torch.utils.data.DataLoader(train_idx,batch_size=100,
                                shuffle=False,sampler=sampler)
    crit_ft1t2=nn.CrossEntropyLoss()
    crit_ft=nn.CrossEntropyLoss()
    crit_t1t2=nn.CrossEntropyLoss()
    crit_f=nn.CrossEntropyLoss()
    crit_t=nn.CrossEntropyLoss()
    optimizer=optim.Adam(classifier.parameters(),weight_decay=0.0)

    val_acc=0.0
    val_classifier=None
    count_val_same=0
    max_epoch=2000
    if not train:
        stop=True
    else:
        stop=False
    epoch=0
    val_classifier=copy.deepcopy(classifier)
    while not stop:  # loop over the dataset multiple times
        running_acc=0.0
        running_loss=0.0
        for i,batch in enumerate(loader_train):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            data=embed_convers(classifier,conversA[:,:,batch,:],
                                conversB[:,:,batch,:],steps_agree[batch],opt,
                                use_agentA,use_agentB)
            f_batchlabels=f_labels[batch]
            t1_batchlabels=t1_labels[batch]
            t2_batchlabels=t2_labels[batch]

            out=classifier(data)

            # TRIPLETS
            if classifier.type_pred==1:
                labels=f_batchlabels*n_t*n_t+t1_batchlabels*n_t+t2_batchlabels
                loss=crit_ft1t2(out,labels)
            elif classifier.type_pred==2:
                labels1=f_batchlabels*n_t*n_t+t1_batchlabels*n_t+t2_batchlabels
                labels2=f_batchlabels*n_t*n_t+t2_batchlabels*n_t+t1_batchlabels
                loss=(crit_ft1t2(out,labels1)+crit_ft1t2(out,labels2))/2.

            # FRUIT + 1 tool
            elif classifier.type_pred==3:
                labels=f_batchlabels*n_t+t1_batchlabels
                loss=crit_ft(out,labels)
            elif classifier.type_pred==4:
                labels=f_batchlabels*n_t+t2_batchlabels
                loss=crit_ft(out,labels)
            elif classifier.type_pred==5:
                labels1=f_batchlabels*n_t+t1_batchlabels
                labels2=f_batchlabels*n_t+t2_batchlabels
                loss=(crit_ft(out,labels1)+crit_ft(out,labels2))/2.

            # FRUIT
            elif classifier.type_pred==6:
                labels=f_batchlabels
                loss=crit_f(out,labels)

            # TUPLE TOOLS
            elif classifier.type_pred==7:
                labels=t1_batchlabels*n_t+t2_batchlabels
                loss=crit_t1t2(out,labels)
            elif classifier.type_pred==8:
                labels1=t1_batchlabels*n_t+t2_batchlabels
                labels2=t2_batchlabels*n_t+t1_batchlabels
                loss=(crit_t1t2(out,labels1)+crit_t1t2(out,labels2))/2.

            # 1 tool
            elif classifier.type_pred==9:
                labels=t1_batchlabels
                loss=crit_t(out,labels)
            elif classifier.type_pred==10:
                labels=t2_batchlabels
                loss=crit_t(out,labels)
            elif classifier.type_pred==11:
                labels1=t1_batchlabels
                labels2=t2_batchlabels
                loss=(crit_t(out,labels1)+crit_t(out,labels2))/2.
            loss.backward()

            if classifier.type_pred in [1,3,4,6,7,9,10]:
                _,pred=torch.max(out,1)
                acc=(pred==labels).nonzero().shape[0]
            elif classifier.type_pred in [2,5,8,11]:
                _,pred=torch.max(out,1)
                acc=((pred==labels1)+(pred==labels2)).nonzero().shape[0]

            running_loss+=loss.item()
            running_acc+=acc
            optimizer.step()

        if epoch>max_epoch:
            stop=True

        if epoch % 50==0:
            print("Epoch %d"%epoch)
            accT=test_classifier(classifier,opt,steps_agree,conversA,conversB,
                train_idx,use_agentA,use_agentB,lookup,lookup_ftool,
                lookup_tools,f_labels,t1_labels,t2_labels)
            train_acc=accT[classifier.type_pred-1]
            print("Training Acc %.2f"%(train_acc))
            print("Running Training Acc %.2f"%(running_acc/Ntrain*100.))
            acc=test_classifier(classifier,opt,steps_agree,conversA,conversB,
                val_idx,use_agentA,use_agentB,lookup,lookup_ftool,lookup_tools,
                f_labels,t1_labels,t2_labels)
            new_val_acc=acc[classifier.type_pred-1]
            print("Validation Acc %.2f"%(new_val_acc))
            if new_val_acc < val_acc:
                stop=True
            elif new_val_acc == val_acc and epoch>0:
                count_val_same+=1
            else:
                val_acc=new_val_acc
                val_classifier=copy.deepcopy(classifier)
                count_val_same=0
            if count_val_same==2:
                stop=True
        epoch+=1

    return val_acc,val_classifier

def test_proba_perf(train_labels,test_labels,n):

    train_size=train_labels.size(0)
    test_size=test_labels.size(0)
    proba=np.zeros(n)
    u_labels, counts=np.unique(train_labels.cpu(),return_counts=True)
    proba[u_labels]=counts/float(train_size)
    perf=proba[test_labels.cpu()].sum()/test_size*100.
    return perf

def max_perf(whole_conv,f_labels,t1_labels,t2_labels,n_t):

    triplet_labels=f_labels*n_t*n_t+t1_labels*n_t+t2_labels
    # whole conv of shape T*2, max_length, N, vocab_size
    u_conv, conv_idx, counts=np.unique(whole_conv,axis=2,
                                        return_counts=True,return_inverse=True)
    N=float(conv_idx.shape[0]) #warning Python 2 return classic division
    conv_idx=torch.as_tensor(conv_idx)
    classes,_=conv_idx.unique().sort()

    f_perf_per_conv=torch.zeros(classes.size(0))
    t1_perf_per_conv=torch.zeros(classes.size(0))
    t2_perf_per_conv=torch.zeros(classes.size(0))
    ft1t2_perf_per_conv=torch.zeros(classes.size(0))
    conv_prob=torch.zeros(classes.size(0))
    for idx_c,c in enumerate(classes):
        #probability of the conversation class c
        c_bool=conv_idx.eq(c)
        conv_prob[idx_c]=c_bool.sum().item()/N

        #number of occurences of most represented label for conversation class c
        #divided by number of labels for conversation class c
        f_perf_per_conv[idx_c]=perf_labels(f_labels,c_bool)
        t1_perf_per_conv[idx_c]=perf_labels(t1_labels,c_bool)
        t2_perf_per_conv[idx_c]=perf_labels(t2_labels,c_bool)
        ft1t2_perf_per_conv[idx_c]=perf_labels(triplet_labels,c_bool)

    # performance is (proba of conv * performance by predicting most represented
    # label)
    f_perf=torch.dot(conv_prob,f_perf_per_conv)*100.
    t1_perf=torch.dot(conv_prob,t1_perf_per_conv)*100.
    t2_perf=torch.dot(conv_prob,t2_perf_per_conv)*100.
    triplet_perf=torch.dot(conv_prob,ft1t2_perf_per_conv)*100.
    return triplet_perf,f_perf,t1_perf,t2_perf

def perf_labels(labels,bool_array):

    u_c, counts_c=np.unique(labels[bool_array],return_counts=True)
    n_c=counts_c.sum()
    max_c=counts_c.max()
    perf=max_c/n_c
    return perf

def embed_convers(classifier,conversA,conversB,steps_agree,opt,\
                                use_agentA,use_agentB):
    device=conversA.device
    hidden=torch.zeros((1,steps_agree.shape[0],classifier.rnn_h_size),
                    device=device)
    for t in range(opt.T):
        if use_agentA:
            batch_idx=(steps_agree>2*t).nonzero()
            if batch_idx.numel()>0:
                batch_idx=batch_idx.squeeze(1)
                batch_idx=batch_idx.tolist()
                h0=hidden[:,batch_idx,:]
                embedded_utt=classifier.embed(conversA[t,:,batch_idx,:],h0)
                # replace with the new embedded utterances
                # if we are to continue, these will be used to initialize
                hidden[:,batch_idx,:]=embedded_utt
        if use_agentB:
            batch_idx=(steps_agree>(2*t+1)).nonzero()
            # replace the 0's with embeddings only if the dialogue is continuing
            if batch_idx.numel()>0:
                batch_idx=batch_idx.squeeze(1)
                batch_idx=batch_idx.tolist()
                h0=hidden[:,batch_idx,:]
                embedded_utt=classifier.embed(conversB[t,:,batch_idx,:],h0)
                hidden[:,batch_idx,:]=embedded_utt
    return hidden.squeeze(0)
