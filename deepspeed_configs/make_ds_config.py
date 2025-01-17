import os
import json
import time

overwrite = True
config_path = "ds_zero3_offlaod_bf16.json"

config = {
#            "zero_optimization": {
#                "stage": 2,
#                #"offload_optimizer": {
#                #    "device": "cpu",
#                #    "pin_memory": True
#                #},
#                "allgather_partitions": True,
#                "allgather_bucket_size": 5e8,
#                "overlap_comm": True,
#                "reduce_scatter": True,
#                "reduce_bucket_size": 5e8,
#                "contiguous_gradients": True,
#                "round_robin_gradients": True
#            },
#
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },

            "optimizer": {
               "type": "AdamW",
               "params": {
                 "lr": "auto",
                 "betas": "auto",
                 "eps": "auto",
                 "weight_decay": "auto"
               }
            },

            #"scheduler": {
            #    "type": "WarmupDecayLR",
            #    "params": {
            #        "total_num_steps": "auto",
            #        "warmup_min_lr": "auto",
            #        "warmup_max_lr": "auto",
            #        "warmup_num_steps": "auto"
            #    }
            #},

            # for fp32
            #"fp16": {
            #    "enabled": false
            #},

            # for fp16
            #"fp16": {
            #    "enabled": "auto",
            #    "loss_scale": 0,
            #    "loss_scale_window": 1000,
            #    "initial_scale_power": 16,
            #    "hysteresis": 2,
            #    "min_loss_scale": 1
            #},

            # for bf16
            "bf16": {
                "enabled": "auto"
            },


            "train_micro_batch_size_per_gpu": "auto",
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",

            "communication_data_type": "fp32",

            "gradient_clipping": "auto",
        }

if os.path.exists(config_path):
    print("New configuraiton path '{}' exists".format(config_path))
    if overwrite:
        print("(WARNING) The existing one will be overwritten in 5 seconds")
        time.sleep(5)
        os.remove(config_path)
    else:
        raise SyntaxError("Provide another name, if you don't want to overwrite")
    
with open(config_path, "w") as json_file:
    json.dump(config, json_file)

print("New configuraiton is saved: {}".format(config_path))

