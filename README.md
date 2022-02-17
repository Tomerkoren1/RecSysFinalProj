# Recommender Systems Final Project

### Installation

- Clone this repo
- please type the command `pip install -r requirements.txt`

### Training
- Run the 'main.py' from ./Code
- The following arguments are supported:
1. --learning_rate - chose learning rate value
2. --epoch_num - epochs number
3. --batch_size - batch size
4. --hidden_num - a list of numbers which define the netwrok structure. For example [1024,512,256] will create 3 hidden layer.
5. --model_name - define the model to train. Support 'BiasMF','AutoRec' and 'OurAutoRec'
6. --dataset - define the dataset to be used for training. Support 'ml-1m' and 'ml-100k'
7. --latent_dim - set the latent dimenstion for the BiasMF model
8. --act_type - set the activation type after FC for OurAutoRec model
9. --dropout - dropout value
10. --momentum - momentum value
11. --weight_decay - weight decay value

### Hyper Parameters
We use the wandb sweep feature to run hyper parameters optimization. To reprduce our work please do the following:
- Create an account for wandb and login on your computer. For more details, see https://wandb.ai/site
- Create a new project
- Create a new sweep, copy one of our sweep configuration (one per model type) and paste it in your sweep configuration. 
Our sweep configuration files located at the 'Sweeps' folder. 
- Open the 'main.py' file and mofify the wand.init entity parameter:
```
wandb.init(project="EnterYourWandbProjectName", entity="EnterYourWandbUserName", config=vars(userArgs))
```
- Launch the sweep agent from the command line
