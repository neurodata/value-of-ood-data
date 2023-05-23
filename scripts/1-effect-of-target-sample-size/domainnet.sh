python3 train.py -m seed=10 reps=3 deploy=True device=cuda:0 tag=1-effect-of-target-sample-size/domainnet task.dataset=domainnet task.target_env=real task.ood_env=quick task.n=5,10,20,50,100 task.m_n=0,1,2,3,4,5,10,20,50,100 task.task_map=[[23,26,34,52,57,67,84,87,94,95,102,106,120,128,132,144,150,162,177,189,194,207,211,215,221,225,238,239,245,256,258,260,261,271,272,284,299,312,336,343]] hp.bs=32 hp.epochs=150 hydra.launcher.n_jobs=8