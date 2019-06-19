# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import one_hot_categorical
from utils import sample_stgs, one_hot
import pdb
import numpy as np

def create_agents(opt):

    # AGENT A
    listenerA=ListenerModuleReinf(opt.vocab_size,opt.symb_embedding_size,
                    opt.rnn_h_size,opt.max_length,opt.EOS_token)
    speakerA=SpeakerModuleReinf(opt.vocab_size,opt.symb_embedding_size,
                    opt.rnn_h_size,opt.max_length,
                    opt.EOS_token)
    proposalGenA=ProposalGenerateModuleReinf(opt.n_choices,opt.features_size)
    im_embedderA=FruitProcessorModule(opt.fruit_feat_size,opt.im_embedding_size)
    tool_embedderA=ToolProcessorModule(opt.n_tools,opt.tool_feat_size,
                    opt.tool_embedding_size)
    bodyA=BodyModule(opt.parsed_input_size,opt.body_features_size,
                        opt.features_size)
    agentA=Agent(listenerA,speakerA,proposalGenA,
                im_embedderA,tool_embedderA,bodyA,opt)
    # AGENT B
    listenerB=ListenerModuleReinf(opt.vocab_size,opt.symb_embedding_size,
                    opt.rnn_h_size,opt.max_length,opt.EOS_token)
    speakerB=SpeakerModuleReinf(opt.vocab_size,opt.symb_embedding_size,
                    opt.rnn_h_size,opt.max_length,
                    opt.EOS_token)
    proposalGenB=ProposalGenerateModuleReinf(opt.n_choices,opt.features_size)
    im_embedderB=FruitProcessorModule(opt.fruit_feat_size,opt.im_embedding_size)
    tool_embedderB=ToolProcessorModule(opt.n_tools,opt.tool_feat_size,
                    opt.tool_embedding_size)

    bodyB=BodyModule(opt.parsed_input_size,opt.body_features_size,
                        opt.features_size)
    agentB=Agent(listenerB,speakerB,proposalGenB,
                    im_embedderB,tool_embedderB,bodyB,opt)
    return agentA, agentB

class BodyModule(nn.Module):

    def __init__(self,parsed_input_size,body_features_size,features_size):
        super(BodyModule, self).__init__()
        self.body_features_size=body_features_size
        self.lin=nn.Linear(parsed_input_size+features_size,features_size,
                    bias=True)

    def forward(self,input_emb,com_emb,prev_state):
        cat_x=torch.cat([input_emb,com_emb,prev_state],dim=-1)
        out=torch.tanh(self.lin(cat_x))
        return out

class Agent(nn.Module):

    def __init__(self,listener,speaker,proposalGen,
                        im_embedder,tool_embedder,body,opt):
        super(Agent, self).__init__()
        self.listener=listener
        self.speaker=speaker
        self.proposalGen=proposalGen
        self.im_embedder=im_embedder
        self.tool_embedder=tool_embedder
        self.body=body
        self.block_mess=opt.block_mess
        self.history=opt.history
        dummy_message=torch.zeros(speaker.max_length,1,speaker.vocab_size)
        dummy_message[:,:,0]=1
        self.dummy_message=dummy_message
        self.h0=torch.zeros(1,self.speaker.rnn_h_size)
        self.h=None
        self.z=None

    def reset_agent(self):
        self.h=None
        self.z=None

    def embed_input(self,images_vectors, tools_vectors):
        if images_vectors is not None:
            input_emb=self.im_embedder(images_vectors)
        # atm we consider each agent receives only 1 input (tool or im)
        else:
            input_emb=self.tool_embedder(tools_vectors)
        self.input_emb=input_emb

    def forward(self,comm_tm1,train=True,id_input=None):

        device=comm_tm1.device
        batch_size=comm_tm1.shape[1]
        if self.block_mess:
            dummy_comm=self.dummy_message.repeat(1,batch_size,1).to(device)
            mess_emb=self.listener(dummy_comm)
        else:
            mess_emb=self.listener(comm_tm1)
        if self.h is None:
            self.h=self.h0.repeat(batch_size,1).to(device)

        z=self.body(self.input_emb,mess_emb,self.h)
        self.z=z
        comm_t,log_pcomm_t=self.speaker(z,train)
        if self.history:
            self.h=self.z
        else:
            self.h=None
        proposal_t,log_pchoice_t=self.proposalGen(z,train)
        return proposal_t,comm_t,log_pchoice_t,log_pcomm_t

class ListenerModuleReinf(nn.Module):
    def __init__(self,vocab_size,embedding_size,rnn_h_size,max_length,EOS_token):
        super(ListenerModuleReinf,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.rnn_h_size=rnn_h_size
        self.max_length=max_length
        self.embeddingTable=nn.Linear(self.vocab_size,self.embedding_size,bias=True)
        self.EOS_token=EOS_token
        self.rnn=nn.RNN(self.embedding_size,self.rnn_h_size,1)

    def forward_rnn(self,x_t,h_tm1,mask_t=None):

        h_t=torch.tanh(self.lin_x(x_t)+self.lin_h(h_tm1))

        if mask_t is not None:
            h_t=h_t*mask_t+h_tm1*(1-mask_t)
        return h_t

    def process_sentence_no_mask(self,received_message,seq_embeddings,h0):

        device=received_message.device
        batch_size=received_message.size(1)
        # rnn input is of size seq_length x batch_size x embedding_size
        output,hidden=self.rnn(seq_embeddings,h0)
        eos_bool=received_message[:,:,self.EOS_token].t()
        mess_length=(eos_bool.cumsum(dim=1).eq(0)).sum(1)
        # If no EOS, then mess_length is max_length
        mess_length=mess_length.clamp(max=self.max_length-1)
        # Take the hidden layer of EOS token or end if no EOS
        h_t=output[mess_length,range(batch_size),:]
        return h_t

    def forward(self,received_message):
        device=received_message.device
        # received_message is of size seq_length x batch_size x vocab_size
        batch_size=received_message.size(1)
        seq_length=received_message.size(0)

        # seq_embeddings is of size batch_size x seq_length x embedding_size
        mess_t=received_message.permute(1,0,2).contiguous().view(batch_size*seq_length,-1)
        seq_embeddingsV=self.embeddingTable(mess_t).view(batch_size,
                                                        seq_length,-1)
        # start hidden state 1 x batch_size x rnn_h_size
        h0=torch.zeros(1,batch_size,self.rnn_h_size,device=device)
        repr_message=self.process_sentence_no_mask(received_message,
                                                seq_embeddingsV.permute(1,0,2),
                                                h0)
        return repr_message

class FruitProcessorModule(nn.Module):
    def __init__(self,im_features_size,im_embedding_size):
        super(FruitProcessorModule,self).__init__()
        self.im_embedding_size=im_features_size
        self.lin=nn.Linear(im_features_size,im_embedding_size,bias=True)

    def forward(self,x):
        out=torch.tanh(self.lin(x))
        return out

class ToolProcessorModule(nn.Module):
    def __init__(self,n_tools,tool_features_size,tool_embedding_size):
        super(ToolProcessorModule,self).__init__()
        self.n_tools=n_tools
        self.tool_features_size=tool_features_size
        per_tool_embedding_size=int(tool_embedding_size/n_tools)
        self.lin=nn.Linear(tool_features_size,per_tool_embedding_size,bias=True)
    def forward(self,x):
        # x is of dim 2 (n_tools) x batch_size x tool_feat_size
        # shared weights
        out1=torch.tanh(self.lin(x[0,:,:]))
        out=torch.cat([out1],dim=-1)
        for t in range(1,self.n_tools):
            out2=torch.tanh(self.lin(x[t,:,:]))
            out=torch.cat([out,out2],dim=-1)
        return out

class ProposalGenerateModuleReinf(nn.Module):
    # proposal is sharing (scalar),tool choice (0/1),termination (binary 0/1)
    def __init__(self,n_choices,features_size):
        super(ProposalGenerateModuleReinf,self).__init__()
        self.n_choices=n_choices
        self.features_size=features_size
        self.lin_tool=nn.Linear(features_size,n_choices,bias=True)

    def forward(self,z,train=True):
        device=z.device
        # z of size batch_size x rnn_h_size
        prop_proba_no_sft=self.lin_tool(z)
        log_p=F.log_softmax(prop_proba_no_sft,dim=-1)
        if train:
            m=torch.distributions.categorical.Categorical(logits=log_p)
            choice_s=m.sample()
        else:
            choice_s=log_p.argmax(-1)
        prop_choice=one_hot(choice_s,self.n_choices)
        sh=0.5*torch.ones(z.size(0),1,device=device)
        proposal=torch.cat([sh,prop_choice.float()],dim=-1)
        return proposal,log_p

class SpeakerModuleReinf(nn.Module):
    def __init__(self,vocab_size,embedding_size,rnn_h_size,
                    max_length,EOS_token):
        super(SpeakerModuleReinf,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.rnn_h_size=rnn_h_size
        self.embeddingTable=nn.Linear(self.vocab_size,self.embedding_size,
                                                                bias=True)
        self.rnn=nn.RNN(self.embedding_size,self.rnn_h_size,1)
        self.lin=nn.Linear(rnn_h_size, vocab_size,bias=True)
        self.max_length=max_length
        self.EOS_token=EOS_token

    def forward(self,z,train):
        device=z.device
        # z of size batch_size x rnn_h_size
        batch_size=z.size(0)
        seq_length=0

        hidden=z.unsqueeze(0)
        # initialization: use Linear + logsumexp + sampling
        proba_no_sft=self.lin(hidden.squeeze(0))
        sequences=[]
        log_p=F.log_softmax(proba_no_sft,dim=-1)
        if train:
            m=torch.distributions.categorical.Categorical(logits=log_p)
            sample=m.sample()
            samp_symb=one_hot(sample,self.vocab_size)
        else:
            sample=log_p.argmax(-1)

        samp_symb=one_hot(sample,self.vocab_size)
        log_p_sent=log_p[range(batch_size),sample]
        binary=samp_symb[:,self.EOS_token]
        mask_t=(1.-binary)
        sequences=[samp_symb.unsqueeze(0)]

        for seq_length in range(1,self.max_length):
            input=self.embeddingTable(samp_symb).unsqueeze(0)
            output,hidden=self.rnn(input, hidden)
            # use last layer and last step hidden
            proba_no_sft=self.lin(hidden.squeeze(0))
            log_p=F.log_softmax(proba_no_sft,dim=-1)
            if train:
                m=torch.distributions.categorical.Categorical(logits=log_p)
                sample=m.sample()
                samp_symb=one_hot(sample,self.vocab_size)
            else:
                sample=log_p.argmax(-1)

            samp_symb=one_hot(sample,self.vocab_size)
            log_p_sent+=mask_t*log_p[range(batch_size),sample]
            binary=samp_symb[:,self.EOS_token]
            mask_t=mask_t*(1.-binary)
            sequences.append(samp_symb.unsqueeze(0))

        sentences=torch.cat(sequences,dim=0)
        return sentences,log_p_sent

    def log_proba_input_sentence(self,z,input_sentence):
        # z of size batch_size x rnn_h_size
        batch_size=z.size(0)
        seq_length=0

        hidden=z.unsqueeze(0)
        # initialization: use Linear + logsumexp + sampling
        proba_no_sft=self.lin(hidden.squeeze(0))
        log_p=F.log_softmax(proba_no_sft,dim=-1)
        # the sample is one-hot
        samp_symb=input_sentence[seq_length,:,:]
        sample=samp_symb.argmax(-1)
        log_p_sent=log_p[range(batch_size),sample]

        binary=samp_symb[:,self.EOS_token]
        mask_t=(1.-binary)
        for seq_length in range(1,self.max_length):
            input=self.embeddingTable(samp_symb).unsqueeze(0)
            output,hidden=self.rnn(input, hidden)
            proba_no_sft=self.lin(hidden.squeeze(0))
            log_p=F.log_softmax(proba_no_sft,dim=-1)

            samp_symb=input_sentence[seq_length,:,:]
            sample=samp_symb.argmax(-1)
            log_p_sent+=mask_t*log_p[range(batch_size),sample]
            binary=samp_symb[:,self.EOS_token]
            mask_t=mask_t*(1.-binary)

        return log_p_sent
