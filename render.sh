#python render.py --heuristic --output "res/LunarLanderHeruictic.gif"
#python render.py --weight "res/weights_lr/3/LunarLander-v2.ckpt" --output "res/LunarLanderAgent1.gif"
#python render.py --weight "res/weights_lr/5/LunarLander-v2.ckpt" --output "res/LunarLanderAgent2.gif"
python render.py --weight "res/last_weight/5/LunarLander-v2.ckpt" --output "res/LunarLanderAgent3.gif" > res/log.txt
