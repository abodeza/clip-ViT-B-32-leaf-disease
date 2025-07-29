# Fine-tuning CLIP on Classifying Plant Diseases
A jupyter notebook based project showcasing the fine-tuning process of CLIP on plant diseases image-disease name pairs.

Under the hood, this is a sentence-transformers model finetuned from sentence-transformers/clip-ViT-B-32. It maps sentences & paragraphs to a None-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

This is a supplementary part of the plant disease detection project found [here](https://abodeza.github.io/projects/gharsa).


## Access Fine-Tuned Model
To access the finetuned model, it can be found [here](https://huggingface.co/abodeza/clip-ViT-B-32-leaf-disease/blob/main/README.md) on Hugging Face.

Or directly used as follows:
```
from sentence_transformers import SentenceTransformer
import torch

device       = "cuda" if torch.cuda.is_available() else "cpu"
clip_model   = SentenceTransformer("abodeza/clip-ViT-B-32-leaf-disease", device=device)
with torch.no_grad():
    text_feats = clip_model.encode(text_prompts, convert_to_tensor=True, device=device)
    text_feats = torch.nn.functional.normalize(text_feats, dim=-1)

    feat       = clip_model.encode([img], convert_to_tensor=True, device=device)
    feat       = torch.nn.functional.normalize(feat, dim=-1)
    sims       = (feat @ text_feats.T).squeeze(0)
```

## Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32) <!-- at revision 11fb331c2c388748c110926aa8013161cb5a85b5 -->
- **Maximum Sequence Length:** 77 tokens
- **Similarity Function:** Cosine Similarity

## Getting Started

### Prerequisites

* Python 3.8+ required
* Required libraries (listed in requirements.txt)
* 7‑Zip is required to unzip the provided data

### Installation

```bash
# Clone the repository
git clone https://github.com/abodeza/clip-ViT-B-32-leaf-disease.git

# Navigate to project directory
cd clip-ViT-B-32-leaf-disease

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Mac: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


### Running the Application

The code provider can be ran on Google Colab or locally and is provided in a jupyter notebook format for ease of personalization. 



## Contributing

Guidelines for contributors:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Author

Abdullah Alzahrani
- Email: [abdullah.alzahrani.p@gmail.com](mailto:abdullah.alzahrani.p@gmail.com)
- GitHub: [@abodeza](https://github.com/abodeza)
- LinkedIn: [Abdullah](https://linkedin.com/in/a-a-alzahrani)
