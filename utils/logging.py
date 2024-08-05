import wandb

def initialize_wandb(path):
    wandb_key = open(path).readlines()[0].strip()
    wandb.login(relogin=True, key=wandb_key)