# pretraining wrn10-2 with 32x32 imagenet
# python3 pretrain.py -m seed=10 deploy=False device=cuda:3 tag=15_pretraining dataset=imagenet net=wrn10_2 task=1 augment=False hp.bs=128 hp.epochs=100

# split cifar10 Ti vs Tj (without pretraining)
# python3 train.py -m seed=10 reps=10 deploy=True device=cuda:3 tag=15_pretraining/wrn10_2 ptw=False task.dataset=split_cifar10 net=wrn10_2 task.custom_sampler=False task.target=1 task.ood=[4] task.m_n=0,1,2,3,4,5,10,20 task.augment=False hp.bs=128 hp.epochs=100 hp.lr=0.01

# split cifar10 Ti vs Tj (with pretraining)
# python3 train.py -m seed=10 reps=10 deploy=True device=cuda:2 tag=15_pretraining/wrn10_2/alt100epochs ptw=True task.dataset=split_cifar10 net=wrn10_2 task.custom_sampler=False task.target=1 task.ood=[4] task.m_n=0,1,2,3,4,5,10,20 task.augment=False hp.bs=128 hp.epochs=100 hp.lr=0.001

# split cifar10 Ti vs Tj (with augmentation)
# python3 train.py -m seed=10 reps=10 deploy=True device=cuda:2 tag=15_pretraining/wrn10_2/aug ptw=False task.dataset=split_cifar10 net=wrn10_2 task.custom_sampler=False task.target=1 task.ood=[4] task.m_n=0,1,2,3,4,5,10,20 task.augment=True hp.bs=128 hp.epochs=100 hp.lr=0.01

#  cifar10 Ti vs Tj (with augmentation + pretraining)
python3 train.py -m seed=10 reps=10 deploy=True device=cuda:1 tag=6-effect-of-pretraining/pretraining-plus-augmentation ptw=True task.dataset=split_cifar10 net=wrn10_2 task.custom_sampler=False task.target=1 task.ood=[4] task.m_n=0,1,2,3,4,5,10,20 task.augment=True hp.bs=128 hp.epochs=100 hp.lr=0.001
