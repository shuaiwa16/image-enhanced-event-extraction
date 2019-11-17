lrs=(7e-6 9e-6 2e-5 4e-5 6e-5) # 2e-5
train_batchs=(8 16 32) # 32
num_train_epochss=(3 4 5 6) # 4

for lr in ${lrs[@]};
do
	for train_batch in ${train_batchs[@]};
	do
		for num_train_epochs in ${num_train_epochss[@]};
		do
			echo "parameter lr:{$lr} train_batch:{$train_batch} num_train_epochs:{$num_train_epochs}"
			if [ ! -d "ace_en/output/output_{$lr}_{$train_batch}_{$num_train_epochs}" ];
			then
				mkdir "ace_en/output/output_{$lr}_{$train_batch}_{$num_train_epochs}"
			else
				echo "already exist"
			fi
			python3 bert_final.py \
			--task_name=ner \
			--do_train=True \
			--do_eval=True \
			--do_predict=True \
			--data_dir="EEdata/" \
			--vocab_file="uncased_L-12_H-768_A-12/vocab.txt" \
			--bert_config_file="uncased_L-12_H-768_A-12/bert_config.json" \
			--init_checkpoint="uncased_L-12_H-768_A-12/bert_model.ckpt" \
			--max_seq_length=32 \
			--train_batch_size=$train_batch \
			--learning_rate=$lr \
			--num_train_epochs=$num_train_epochs \
			--output_dir="ace_en/output/output_{$lr}_{$train_batch}_{$num_train_epochs}/" \
			> "ace_en/result/result_{$lr}_{$train_batch}_{$num_train_epochs}.txt"
		done
	done
done


# python3 bert_ws.py \
# 	--task_name=ner \
# 	--do_train=True \
# 	--do_eval=True \
# 	--do_predict=True \
# 	--data_dir="EEdata/" \
# 	--vocab_file="uncased_L-12_H-768_A-12/vocab.txt" \
# 	--bert_config_file="uncased_L-12_H-768_A-12/bert_config.json" \
# 	--init_checkpoint="uncased_L-12_H-768_A-12/bert_model.ckpt" \
# 	--max_seq_length=64 \
# 	--train_batch_size=32 \
# 	--learning_rate=2e-5 \
# 	--num_train_epochs=4 \
# 	--output_dir="ace_en/output/" \
# 	> "result.txt"
