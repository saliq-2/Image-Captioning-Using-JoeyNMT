# Image-Captioning-Using-JoeyNMT





###**Demo**







<img width="552" height="408" alt="vgg_16_output" src="https://github.com/user-attachments/assets/b2a5bc36-6afb-485d-9ad5-fab850e8f429" />











### Training

1) Install dependencies
pip install -r requirements.txt


2) Prepare dataset
- Load your dataset (images + captions).
- Expected format:
  - `captions_file`: JSON mapping image filename → list of captions, e.g. `{ "img1.jpg": ["a dog runs", ...], ... }`
  - `images_dir`: folder containing all images referenced in the JSON.
- Place them anywhere on disk.

3) Configure paths and parameters
- Open `config.py` and set:
  - `captions_file` to your JSON file
  - `images_dir` to your images folder
  - optional: `encoder` (`vgg16` or `resnet50`), `batch_size`, `num_epochs`, `learning_rate`, etc.

4) Start training

python main.py


5) Monitoring and artifacts
- Training loss, perplexity, and periodic BLEU/METEOR are printed to console.
- A plot of training curves is saved as `comprehensive_training_curves.png`.
- The best model checkpoint is saved to `save_path` from `config.py` (default: `best_image_captioning_model.pth`).

6) Inference (single image)

python predict.py /abs/path/to/image.jpg "/abs/path/to/checkpoint.pth" --max_len 20


Notes
- Wrap paths with spaces in quotes.
- GPU is used automatically if available.
