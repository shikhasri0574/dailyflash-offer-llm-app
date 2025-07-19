# 🚀 DailyFlash Offer Generator

A lightweight LLM-based solution for generating structured JSON offers and promotional images from raw promotional text. This project uses a fine-tuned GPT2 model to transform unstructured promotional content into standardized JSON format, and Stable Diffusion for creating promotional images.

## 📋 Project Overview

This project demonstrates how to:
- Fine-tune a GPT2 model on custom data using Google Colab (free tier)
- Structure input-output pairs for training
- Process and generate JSON from raw text
- Generate promotional images from structured offer data
- Create a full UI for testing and demonstration

## 🗂️ Project Structure

```
dailyflash-offer-llm-app/
├── data/
│   └── offer_dataset.jsonl         # Training data with 50 examples
├── train_model_colab.ipynb         # Colab notebook to train GPT2
├── generate_offer_image_colab.ipynb # Colab notebook for image generation
├── generate_offer.py               # Inference script
├── gradio_ui.py                    # Gradio-based UI for text generation
├── gradio_ui_with_images.py        # Enhanced UI with image generation
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

## 🔧 Setup & Installation

1. **Clone this repository**:
   ```bash
   git clone <repository-url>
   cd dailyflash-offer-llm-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   - Open `train_model_colab.ipynb` in Google Colab
   - Follow the instructions in the notebook to upload the dataset and train the model
   - Save the trained model to Google Drive or download it to your local machine

4. **Run the Gradio UI**:
   ```bash
   # For text-only generation:
   python gradio_ui.py
   
   # For text and image generation:
   python gradio_ui_with_images.py
   ```

## 🚆 Training the Model

The training process is documented in detail in the `train_model_colab.ipynb` notebook. Here's a summary:

1. Upload the dataset (`data/offer_dataset.jsonl`) to Colab
2. Load and preprocess the data
3. Fine-tune a GPT2 model using the HuggingFace Transformers library
4. Save the model to Google Drive or download it

## 🧪 Using the Model

### Command Line Interface

```bash
python generate_offer.py --model_path "path/to/model" --input "Buy 2 get 1 free on shirts at ABC Store, Lucknow."
```

### Python API

```python
from generate_offer import load_model

# Load the model
generator = load_model("path/to/model")

# Generate an offer
result = generator.generate_offer("50% off on all electronics at TechMart until Dec 25.")
print(result)
```

### Web UI

The project includes two Gradio-based web interfaces for easy testing:

```bash
# Basic UI (text-only)
python gradio_ui.py

# Enhanced UI with image generation
python gradio_ui_with_images.py
```

Either command will launch a web server at http://localhost:7860 with a user-friendly interface.

#### Image Generation

The enhanced UI allows you to generate promotional images based on the structured offer data. The system uses prompts derived from the offer JSON to create images that match the content and style of the promotion.

### Using the Text-to-Image Feature

1. **Option 1: Using the Colab Notebook**
   - Open `generate_offer_image_colab.ipynb` in Google Colab
   - Run all cells to set up the environment
   - Enter your offer JSON in the interactive form
   - Click "Generate Offer Image"
   - Download the generated image directly or through the zip archive

2. **Option 2: Using the Enhanced Gradio UI**
   - Run `python gradio_ui_with_images.py`
   - Enter promotional text and click "Generate Offer"
   - Enable the "Generate Promotional Image" checkbox
   - The image will be displayed alongside the JSON offer
   - The prompt used for generation is also shown for reference

3. **Customizing Image Generation**
   - The system selects templates based on the offer category
   - Fashion offers include clothing and seasonal elements
   - Food offers show appetizing imagery and restaurant themes
   - Electronics offers use tech-themed digital elements
   - Other categories use a general promotional template

## 📊 Dataset Format

The dataset follows a simple format where each line is a JSON object with a `text` field containing both input and expected output:

```json
{
  "text": "Input: [raw promotional text]\nOutput: {\"title\": \"...\", \"subtitle\": \"...\", ...}"
}
```

The JSON schema for offers includes:
- `title`: Main offer title
- `subtitle`: Secondary text/headline
- `location`: Where the offer is valid
- `contact`: Contact information
- `category`: Type of product/service
- `discount_type`: Nature of the discount (percentage, fixed, etc.)
- `expiry_date`: When the offer ends
- Plus additional fields as needed (promo_code, condition, etc.)

## 💳 API Keys & Requirements

### Free Implementation (As Provided)
- **No API keys required** for any part of the project
- GPU access on Google Colab for model training and image generation
- Python 3.7+ with dependencies installed via `requirements.txt`

### Alternative Implementations
For production use, you might consider these alternatives that would require API keys:

- **Text Generation**: OpenAI API or similar LLM provider
- **Image Generation**: 
  - Replicate API (offers Stable Diffusion as a service)
  - Stability AI API (for DALL-E or other image models)
  - Playground AI API

## 📈 Performance & Limitations

- The model is trained on a small dataset (50 examples) and may require more data for better performance
- Being based on GPT2-small, it has limitations in handling very complex outputs
- Temperature parameter can be adjusted to balance between creativity and accuracy
- Image generation requires significant computational resources when using Stable Diffusion
- The Colab notebook for image generation uses free tier GPU resources and may have time limitations
- For production use, consider using a dedicated API for image generation

## 📜 License

This project is available under the MIT License.

## 🙏 Acknowledgements

- HuggingFace for the Transformers library
- Gradio for the simple UI framework
