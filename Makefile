install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	-cml comment create report.md || echo "✓ Report generated successfully"

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	@echo "Setting up deployment environment..."
	python -m pip install -U huggingface_hub
	@echo "Logging into Hugging Face..."
	python -m huggingface_hub.cli login --token $(HF) --add-to-git-credential

push-hub:
	@echo "Uploading files to Hugging Face Hub..."
	python - <<'PY'
	from huggingface_hub import upload_folder
	repo_id = "Hafsa7/iris_Classification"
	upload_folder(
	    repo_id=repo_id,
	    repo_type="space",
	    folder_path="./App",
	    commit_message="Sync App files"
	)
	upload_folder(
	    repo_id=repo_id,
	    repo_type="space",
	    folder_path="./Model",
	    path_in_repo="Model",
	    commit_message="Sync Model"
	)
	upload_folder(
	    repo_id=repo_id,
	    repo_type="space",
	    folder_path="./Results",
	    path_in_repo="Results",
	    commit_message="Sync Metrics"
	)
	print("✔ Upload finished successfully.")
	PY

deploy: hf-login push-hub
