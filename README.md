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
