#!/bin/sh

cd externals/tensorflow_models/research/audioset

# Download data files into same directory as code.
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
cd -

mkdir data
mkdir data/audioset
cd data/audioset

# Download csv data files
curl -O http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
curl -O http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
curl -O http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv
curl -O http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/qa_true_counts.csv
curl -O http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/rerated_video_ids.txt

# Download the feature embeddings
curl -O http://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz
tar -xzf features.tar.gz

cd -