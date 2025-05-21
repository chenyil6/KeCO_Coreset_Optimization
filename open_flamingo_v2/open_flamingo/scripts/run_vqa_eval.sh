DEVICE=0 # gpu num

RANDOM_ID="VQAV2_RS"
RESULTS_FILE="${RANDOM_ID}.json"

nohup python -u open_flamingo/eval/evaluate_siir.py \
    --model "open_flamingo" \
    --lm_path "/path/to/mpt-7b/" \
    --lm_tokenizer_path "/path/to/mpt-7b/" \
    --checkpoint_path "/path/to/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt" \
    --vision_encoder_path "ViT-L-14" \
    --vision_encoder_pretrained 'openai' \
    --device $DEVICE \
    --vqav2_train_image_dir_path "/path/to/coco/train2014/"  \
    --vqav2_train_questions_json_path "/path/to/vqav2/" \
    --vqav2_train_annotations_json_path  "/path/to/data/" \
    --vqav2_test_image_dir_path "/path/to/data/coco/val2014/" \
    --vqav2_test_questions_json_path "/path/to/" \
    --vqav2_test_annotations_json_path "/path/to/" \
    --results_file $RESULTS_FILE \
    --num_samples 5 \
    --shots 2 \
    --num_trials 1 \
    --batch_size 1 \
    --cross_attn_every_n_layers 4 \
    --precision fp16 \
    --eval_vqav2  > RECORD_VQAV2_RS.log 2>&1 &


echo "evaluation complete! results written to ${RESULTS_FILE}"