# Prime Intellect Setup And Test Run

This is a short checklist to prepare the environment and run the smoke test.

## 1) Mount persistent storage

On Prime Intellect, attach your persistent volume to this mount point:

- /mnt/storage

If your runtime does not auto-create it, run:

	sudo mkdir -p /mnt/storage

Quick verification:

	df -h | grep /mnt/storage || true
	ls -lah /mnt/storage

Expected directories for this project:

- /mnt/storage/embeddings
- /mnt/storage/outputs

Create them if needed:

	mkdir -p /mnt/storage/embeddings /mnt/storage/outputs

## 2) Download data with rclone (if needed)

If the embeddings are not already in persistent storage, sync them in with rclone.

Install rclone (if missing):

	curl https://rclone.org/install.sh | sudo bash

Configure your remote once:

	rclone config

Example sync command (replace REMOTE and path):

	rclone copy REMOTE:your/path/to/embeddings /mnt/storage/embeddings -P

Required files in /mnt/storage/embeddings:

- embeddings_train.npy
- embeddings_test.npy
- labels_train.npy
- labels_test.npy

Check:

	ls -lah /mnt/storage/embeddings

## 3) Install uv

Install uv:

	curl -LsSf https://astral.sh/uv/install.sh | sh

Load it into PATH for the current shell:

	export PATH="$HOME/.local/bin:$PATH"

Verify:

	uv --version

## 4) Run the smoke test script

From the repo root, run:

	chmod +x scripts/basic_test.sh
	./scripts/basic_test.sh \
	  --data-dir /mnt/storage/embeddings \
	  --out-dir /mnt/storage/outputs \
	  --smoke-epochs 5 \
	  --project .

Notes:

- Logs are saved automatically under /mnt/storage/outputs.
- Use --no-log if you only want console output.

## 5) Troubleshooting

- uv not found: re-run export PATH="$HOME/.local/bin:$PATH"
- Missing data file error: confirm the four .npy files are in /mnt/storage/embeddings
- pyproject not found warning: run from repo root and pass --project .

