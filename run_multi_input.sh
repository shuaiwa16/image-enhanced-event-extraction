rm -rf _temp/*
python3 BERT_NER.py \
	--task_name=ner \
	--do_train=True \
	--do_eval=True \
	--do_predict=False \
	--data_dir="EEdata/" \
	--vocab_file="wwm_uncased_L-24_H-1024_A-16/vocab.txt" \
	--bert_config_file="wwm_uncased_L-24_H-1024_A-16/bert_config.json" \
	--init_checkpoint="wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt" \
	--num_train_epochs=3 \
	--output_dir="_temp/" \
	> "result.txt"
