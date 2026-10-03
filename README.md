# Restoration and Sketch Studio

Generative AI, Assignment 1. Four generative models behind one web application. Three image restoration systems are trained on the Oxford-IIIT Pet dataset, and a conditional GAN generates sketches from face photos using the FS2K dataset.

## Links

* Repository: https://github.com/AbdullahRasheed452/GENAI-1
* Trained ONNX models on Google Drive: https://drive.google.com/drive/folders/1O5NRmpWf2LKDXC6hxI7BO0HMAYznd3yv
* Technical report: submitted as a PDF on Google Classroom
* Demo video on YouTube: https://youtu.be/kPPl9NkhYq4

## The four tasks

| Workspace | Method | Data |
| --- | --- | --- |
| Universal Restoration | One convolutional autoencoder with a compressed latent vector restores clean, noisy, blurred, and occluded images | Oxford-IIIT Pet, 128 x 128 |
| Hard-Routed Restoration | A CNN classifier names the corruption, then one of three specialist autoencoders restores the image. Clean images bypass restoration | Oxford-IIIT Pet |
| Soft Mixture-of-Experts | A gate gives a weight to the identity branch and the three experts, trained jointly. The output is the weighted sum | Oxford-IIIT Pet |
| Face-to-Sketch Generator | A style-conditioned U-Net generator with a PatchGAN discriminator, with three sketch styles | FS2K, 128 x 128 |

## Quick start (Docker)

You need Git and Docker Desktop. Start Docker Desktop and wait until it reports that the engine is running.

1. Clone the repository.

```
git clone https://github.com/AbdullahRasheed452/GENAI-1.git
cd GENAI-1
```

2. Download the seven ONNX models (about 520 MB) into `models_onnx`.

```
python -m pip install gdown
python scripts/download_models.py
```

If the script fails, for example because Google Drive limited the downloads, open the Drive link above, download the seven `.onnx` files by hand, and place them in the `models_onnx` folder. The expected files are listed below.

3. Start the whole application with one command.

```
docker compose up --build
```

The first build downloads base images and installs packages, which takes a few minutes depending on the connection. The backend then loads the models in about ten seconds and the frontend starts once the backend reports healthy.

4. Open http://localhost:8080 in a browser. The header shows a green badge when the backend is connected and all seven models are loaded. The interactive API documentation is at http://localhost:8000/docs.

5. Stop the application with Ctrl+C in the terminal, then run `docker compose down`.

Expected files in `models_onnx`:

```
task1_universal_ae.onnx
task2_classifier.onnx
task2_specialist_salt_pepper.onnx
task2_specialist_blur.onnx
task2_specialist_occlusion.onnx
task3_soft_moe.onnx
task4_generator.onnx
```

The models are mounted into the backend container as a read-only volume, so they are not part of the image and are not stored in Git.

## Using the application

* Restoration workspaces: upload an image, choose a corruption (none, salt-and-pepper noise, Gaussian blur, rectangular occlusion) and a severity (low, medium, high), then click Restore image. The severities are the three fixed test levels from the assignment. Choose None to upload an image that is already corrupted. The page shows the original, the corrupted input, the restored output, the settings used, the inference time, and PSNR against the original. The button for a different random pattern changes the noise seed or the rectangle placement.
* Hard-Routed workspace: also shows the four classifier probabilities, the predicted corruption, the selected expert, and separate classifier and expert timings.
* Soft Mixture workspace: also shows the four routing weights and highlights the dominant branch.
* Face-to-Sketch workspace: upload a photo or capture one with the webcam, choose Style 1, 2, or 3, and click Generate sketch. The photo and sketch are shown side by side with a download link. The webcam needs browser permission.
* Images are validated on the server: JPEG, PNG, or WebP, at most 10 MB. They are converted to RGB and resized to 128 x 128, the same preprocessing used in training.

## Backend endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /api/health | Lists loaded and missing models |
| POST | /api/universal | Task 1 |
| POST | /api/hard | Task 2 |
| POST | /api/soft | Task 3 |
| POST | /api/sketch | Task 4 |

## Repository layout

```
backend/            FastAPI service and Dockerfile
frontend/           React and Tailwind application, nginx config, Dockerfile
docker-compose.yml  Starts backend and frontend
models_onnx/        ONNX models (downloaded, not in Git)
scripts/            download_models.py
src/data/           dataset download, splits, corruption functions, manifests, FS2K preparation
src/models/         autoencoder, classifier, mixture of experts, conditional GAN
src/training/       losses and training scripts
src/optuna_studies/ Optuna studies for every task
src/evaluation/     metrics, evaluation scripts, report asset generation
src/export/         ONNX export and verification
configs/            fixed dataset split and the validation and test corruption manifests
results/            evaluation tables, figures, Optuna trial tables, best parameters
experiments/mlflow/ MLflow tracking databases
docs/stitch/        Google Stitch design screenshots
report/             figures and LaTeX tables used in the report
```

## Reproducing the training

Training ran on free GPU notebooks (Google Colab and Kaggle). Install `requirements.txt`. The local machine only needs the data scripts and the application. All commands run from the repository root. In the commands below, D1 to D4 stand for output folders of your choice.

Data for Tasks 1 to 3. The split uses seed 42 and the corruption manifests are fixed files in `configs/`.

```
python -m src.data.download_pets
python -m src.data.make_split
python -m src.data.make_manifests
python -m src.data.build_cache
```

Task 1.

```
python -m src.optuna_studies.task1_study --storage_dir D1 --n_trials 20 --epochs 10
python -m src.training.train_final_task1 --storage_dir D1 --epochs 30
python -m src.evaluation.evaluate_restoration --checkpoint D1/ae_universal_final.pt --out_dir D1/eval --name task1
python -m src.export.export_ae --checkpoint D1/ae_universal_final.pt --out D1/task1_universal_ae.onnx
```

Task 2. Needs the Task 1 data steps. D2 is one folder holding the classifier and the specialists.

```
python -m src.optuna_studies.task2_classifier_study --storage_dir D2 --n_trials 15 --epochs 8
python -m src.training.train_classifier --from_study D2 --epochs 30
python -m src.evaluation.evaluate_classifier --checkpoint D2/classifier_final.pt --out_dir D2/eval
python -m src.optuna_studies.task2_specialists_study --storage_dir D2 --n_trials 10 --epochs 6
python -m src.training.train_specialists_final --storage_dir D2 --epochs 40
python -m src.evaluation.evaluate_routed --classifier D2/classifier_final.pt --specialist_dir D2 --out_dir D2/eval
python -m src.export.export_task2 --task2_dir D2 --out_dir D2/onnx
```

Task 3. Needs the trained Task 2 classifier and specialists in D2.

```
python -m src.optuna_studies.task3_study --task2_dir D2 --storage_dir D3
python -m src.training.train_moe --task2_dir D2 --storage_dir D3 --from_study --warmup_epochs 2 --epochs 15
python -m src.evaluation.evaluate_task3 --checkpoint D3/moe_final.pt --out_dir D3/eval
python -m src.export.export_moe --checkpoint D3/moe_final.pt --out D3/onnx/task3_soft_moe.onnx
```

Task 4. Download FS2K from the Google Drive link in the FS2K repository (https://github.com/DengPingFan/FS2K), unzip it, and point the preparation script at the folder that holds `anno_train.json`, `anno_test.json`, `photo`, and `sketch`. The official train and test split comes from those annotation files.

```
python -m src.data.prepare_fs2k --root PATH_TO_FS2K
python -m src.optuna_studies.task4_study --storage_dir D4 --n_trials 12 --epochs 8
python -m src.training.train_cgan --from_study D4 --epochs 60 --sample_every 10
python -m src.evaluation.evaluate_cgan --checkpoint D4/cgan_final.pt --out_dir D4/eval
python -m src.export.export_cgan --checkpoint D4/cgan_final.pt --out D4/task4_generator.onnx
```

Report tables and figures from the saved results:

```
python -m src.evaluation.make_report_assets
```

## Experiment tracking

Every training run and every Optuna trial is logged with MLflow (hyperparameters, losses, validation metrics, and for Task 4 the sample images). The tracking databases are in `experiments/mlflow/`. To browse one in a web page:

```
python -m mlflow ui --backend-store-uri sqlite:///experiments/mlflow/task1_mlflow.db
```

Then open http://127.0.0.1:5000. Use the other database files in the same folder for Tasks 2, 3, and 4. Optuna trial tables and best parameters for all studies are in `results/` as CSV and JSON files.

## Results on the fixed test manifests

Restoration results on 36,690 test entries (3,669 test images, each with one clean version and three severities of three corruptions). The Input row is the score of returning the corrupted image unchanged. Values are PSNR in dB and SSIM, averaged over all corrupted entries.

| Method | PSNR | SSIM |
| --- | --- | --- |
| Input, no restoration | 19.70 | 0.639 |
| Task 1, universal autoencoder | 18.55 | 0.487 |
| Task 2, hard routing with the classifier | 18.44 | 0.494 |
| Task 3, soft mixture of experts | 21.54 | 0.692 |

The Task 2 classifier reaches 0.9980 accuracy and 0.9967 macro F1 on the test entries. Face-to-sketch results on the 1,045 official test pairs: SSIM 0.459 and PSNR 15.48 dB overall, with Style 1 at 0.500, Style 2 at 0.377, and Style 3 at 0.585 SSIM. Full per-condition and per-severity tables, figures, and failure cases are in the report and in `results/`.

## Known limitations

* The autoencoders pass every image through a small latent vector and use no skip connections, so reconstructions are soft. Tasks 1 and 2 score below the unmodified input on average, and blurred images are restored worse than they arrive. The mixture of experts in Task 3 learns to route most blurred images to the identity branch, and its blur expert is effectively inactive.
* The Optuna studies are small because of the time budget, and each result comes from a single training run with one seed.
* The FS2K test set has 1,045 pairs, not 1,046, because one file is missing from the downloaded copy. Style 3 has only 46 test pairs, so its scores are noisy.
* Inference runs on CPU with ONNX Runtime. Typical latency is a few tens of milliseconds per image.

## Troubleshooting

* The header shows missing models: the files in `models_onnx` are absent or misnamed. Run the download script again and restart the containers.
* Port 8080 or 8000 is in use: stop the other program, or change the left side of the port mappings in `docker-compose.yml`.
* Docker cannot connect to the engine: start Docker Desktop and wait for it to report that the engine is running.
* The webcam button fails: allow camera access for localhost in the browser.

## Data and credits

Oxford-IIIT Pet dataset by Parkhi, Vedaldi, Zisserman, and Jawahar. FS2K facial sketch dataset by Fan and coauthors. The use of AI assistants is described in the appendix of the technical report.
