echo "=== Step 1: Preprocess Images (with Augmentation) ==="; `
python scripts/preprocess.py --data_dir data --save_npz outputs/preprocessed_augmented.npz --augment --augment_factor 3; `
echo "=== Step 2: Extract Features (ViT Embeddings) ==="; `
python scripts/extract_features.py --npz_input outputs/preprocessed_augmented.npz --out_npz outputs/features_augmented.npz --batch_size 8 --num_threads 6; `
echo "=== Step 3: Train Classifier (Balanced SVM & Save Metrics) ==="; `
python scripts/train_classifier.py --features outputs/features_augmented.npz --out_model outputs/classifier_final.joblib --metrics_dir outputs/metrics; `
echo "=== Step 4: Test Single Image ==="; `
python scripts/infer.py --image "data/happy/0.jpg" --classifier outputs/classifier_final.joblib; `
echo "=== Step 5: Batch Inference ==="; `
python -m scripts.batch_infer --data_dir data --classifier outputs/classifier_final.joblib --out_csv outputs/predictions_final.csv; `
echo "=== Step 6: Launch Gradio Web App ==="; `
python app.py