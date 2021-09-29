CUDA_VISIBLE_DEVICES=$1 python3 train.py \
	--model_dir "bert-base-uncased" \
	--train_dir "./data/mirror_corpus/en_words_top_10k.txt" \
	--output_dir tmp/bert_base_mirror_en_10k_word_infonce0.2_maxlen25_bs200_mask0_dropout0.1_drophead0.0_cls \
	--epoch 2 \
	--train_batch_size 200 \
	--learning_rate 2e-5 \
	--max_length 25 \
	--infoNCE_tau 0.2 \
	--dropout_rate 0.1 \
	--drophead_rate 0.0 \
	--random_span_mask 0 \
	--agg_mode "cls" \
	--amp \
	--parallel \
	--use_cuda 

