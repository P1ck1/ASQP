python main.py --task asqp \
            --dataset zh \
            --model_name_or_path t5-chinese \
            --n_gpu 1 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20
