python train.py \
    -tr "/Users/gabriel.t.nishimura/projects/masters/datasets/LibriSpeech/librivox-train-clean-simple.csv" \
    -v "/Users/gabriel.t.nishimura/projects/masters/datasets/LibriSpeech/librivox-dev-clean.csv" \
    -te "/Users/gabriel.t.nishimura/projects/masters/datasets/LibriSpeech/librivox-test-clean.csv" \
    --units=512 --batch_size=32 --epoch_len=256 --epochs=3