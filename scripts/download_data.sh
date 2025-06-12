#!/bin/bash

cd ..

KAGGLE_DATASET="adarshsng/googlenewsvectors"
TARGET_DIR="./models"

mkdir -p "$TARGET_DIR"

kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR"

if [ -f "$TARGET_DIR/word2vec-google-news-300.zip" ]; then
    unzip "$TARGET_DIR/word2vec-google-news-300.zip" -d "$TARGET_DIR"
    rm "$TARGET_DIR/word2vec-google-news-300.zip"
fi

echo "Download and extraction complete!"
