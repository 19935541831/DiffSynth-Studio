from diffsynth.core.loader import hash_model_file
 

model_hash = hash_model_file("/project/peilab/Puxin/Wan_action/checkpoints/Wan2.1-I2V-14B-480P/action_encoder.pth")
print(f"hash: {model_hash}")