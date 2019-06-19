# fruit-tools-game
fruits-tools-game

# Fruits and Tools game 

This repository implements the paper "Miss Tools and Mr Fruit: Emergent communication in agents learning about object affordances" (Bouchacourt and Baroni, ACL 2019) (https://arxiv.org/abs/1905.11871).

## Run the code 
```
python train.py  --manualSeed "0" --variance "0.1" --block_mess "0" --vocab_size "10" --history "1" --norm_r "2" --max_length "1" --gc "0.1" --symmetric "1" --batch_size "128" --start_fruit "2" --data_dir "where/you/saved/the/data"
```

Note, the model checkpointing every 100 epochs in train.py (see below) can quickly overload the storage capacity, feel free to save models more rarely.

## Parameters
```
  --cuda # enables cuda
  --ngpu # number of GPUs to use
  --ncpu # number of CPUs to use
  --outf #folder to output images and model checkpoints
  --modeldir # folder to get model checkpoints for testing
  --data_dir # folder to get data
  --manualSeed # manual seed
  --workers # number of data loading workers
  --variance # variance data sampling
  --gc # gradient clipping value
  --lr # learning rate
  --vocab_size # vocabulary size
  --max_length #max sentence length
  --batch_size # training batch size
  --val_batch_size # validation batch size
  --rnn_h_size # rnn hidden layer size
  --input_embedding_size # input embedding size
  --symb_embedding_size # vocab symbols embedding size',
  --body_features_size # body intermediate linear layer size
  --n_episodes # number of episodes in training
  --n_episodes_val # number of episodes in validation
  --T # max number of rounds in each episode
  --norm_r # type of reward : 0 is the utility (scalar) / 1 is utility of chosen tool divided by utility of better tool (scalar between 0 and 1) / 2 is a 1 (or 0) if better tool is chosen (or not). We used 2 in the paper.
  --start_fruit # start the episode with fruit agent (0 is start with Tool, 1 is start with Fruit, 2 is random each Fruit or Tool can start, we used 2 in the paper)
  --min_r # constant added to utility
  --block_mess # message channel is blocked
  --history # feed previous agent hidden state (memory)
  --n_tools # number of tools fed to Tool agent (2 in the paper)
  --symmetric # switch tool / fruit roles : 0 is fixed roles, 1 is random roles i.e. each agent can be either Tool or Fruit at each episode. We use 1 in the paper.
  --corruptA # corrupt messages from agent A
  --corruptB # corrupt messages from agent B
  --sample_test # sample instead of argmax at test time
  --st # use self-talk
  ```
  
## Licence
EGG is licensed under the MIT license. The text of the license can be found [here](LICENSE).
