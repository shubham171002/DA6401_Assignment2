import wandb
from sweep_config import sweep_config
from train_model import train_model

def sweep_train():
    run = wandb.init(project="inaturalist-resnet50-finetuning")
    config = wandb.config
    run.name = f"resnet50_layer4_{config.optimizer}_bs{config.batch_size}_lr{config.learning_rate:.0e}"
    train_model()
    wandb.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="inaturalist-resnet50-finetuning")
    wandb.agent(sweep_id, function=sweep_train, count=20)
