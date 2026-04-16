_base_ = ["./cls-ptv3-v1m1-0-base.py"]

batch_size = 16
num_worker = 8
batch_size_val = 8
enable_wandb = False

epoch = 120
eval_epoch = 20

model = dict(
    backbone=dict(
        order=("z", "z-trans"),
    )
)

test = dict(type="ClsVotingTester", num_repeat=10)
