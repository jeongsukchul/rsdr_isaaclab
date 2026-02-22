wandb_project=factory-peginsert
for seed in 0 1 2 3 4 5
do
    python scripts/rl_games/train.py --task Custom-Factory-PegInsert-Direct-GMMVI-v0 --beta -10 --headless --track --seed $seed --wandb-project-name $wandb_project --success-threshold .6
done 