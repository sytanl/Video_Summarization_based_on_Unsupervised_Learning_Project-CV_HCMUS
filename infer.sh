#!/bin/bash

# Check if video folder path and output folder path are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 video_folder_path output_folder_path"
    exit 1
fi

# Create summary folders inside output folder
mkdir -p "$2/summaries"

# Run summarization script with arguments
python scripts/summarization.py \
--embedding-folder "$2/embeddings" \
--context-folder "$2/contexts" \
--summary-folder "$2/summaries" \
--reduced-emb \
--scoring-mode "uniform" \
--kf-mode "middle ends" \
--k 8