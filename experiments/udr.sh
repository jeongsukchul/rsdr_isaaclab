wandb_project=factory-peginsert-fix
for seed in 0 1 2 3 4 5
do
    python scripts/rl_games/train.py --task Custom-Factory-PegInsert-Direct-UDR-v0 --headless --track --seed $seed --wandb-project-name $wandb_project
done 