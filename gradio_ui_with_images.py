"""
Enhanced Gradio UI for the DailyFlash Offer Generator.
This script creates a web interface for generating structured JSON offers and promotional images.
"""

import os
import json
import gradio as gr
from generate_offer import load_model
import re
from PIL import Image
import tempfile
import time

# Path to the model (you'll need to update this with the actual path after training)
DEFAULT_MODEL_PATH = "model"

def format_json(json_str):
    """
    Format JSON string for better display in the UI
    """
    try:
        # Try to parse JSON
        json_obj = json.loads(json_str)
        # Return formatted JSON
        return json.dumps(json_obj, indent=2)
    except json.JSONDecodeError:
        # If not valid JSON, try to extract JSON using regex
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                json_obj = json.loads(json_str)
                return json.dumps(json_obj, indent=2)
            except json.JSONDecodeError:
                return json_str
        return json_str

def generate_image_from_json(json_str):
    """
    Generate an image based on the offer JSON
    For demonstration purposes, this creates a placeholder image
    In production, you would replace this with calls to Stable Diffusion or other image models
    """
    try:
        # Parse the JSON
        json_obj = json.loads(json_str)
        
        # Extract key information from the offer
        title = json_obj.get('title', '')
        subtitle = json_obj.get('subtitle', '')
        category = json_obj.get('category', 'Retail').lower()
        
        # Create highlights
        highlights = []
        
        # Add discount info if available
        discount_type = json_obj.get('discount_type', '')
        if discount_type == 'percentage' and 'discount_value' in json_obj:
            highlights.append(f"🔥 {json_obj['discount_value']}% OFF")
        elif discount_type == 'bundle':
            highlights.append(f"🎁 {subtitle}")
        
        # Add expiry date if available
        expiry = json_obj.get('expiry_date')
        if expiry:
            highlights.append(f"⏰ Offer valid till {expiry}")
        
        # Add promo code if available
        promo_code = json_obj.get('promo_code')
        if promo_code:
            highlights.append(f"🏷️ Use Code: {promo_code}")
            
        # Build location and contact
        location = json_obj.get('location', '')
        contact = json_obj.get('contact', '')
        store_info = ""
        if location:
            store_info = f"{title.split(' ')[0]}, {location}"
            if contact:
                store_info += f" ({contact})"
        
        # Build prompt for image generation
        prompt_parts = []
        prompt_parts.append(f"Create a vibrant promotional poster for a {category} offer.")
        prompt_parts.append(f"Title: \"{title}\"")
        prompt_parts.append(f"Subtitle: \"{subtitle}\"")
        
        # Add highlights if any
        if highlights:
            prompt_parts.append("Highlights:")
            for highlight in highlights:
                prompt_parts.append(f"- {highlight}")
        
        # Add call to action
        prompt_parts.append(f"Call to Action: \"Shop Now!\"")
        
        if store_info:
            prompt_parts.append(f"Store: {store_info}")
            
        # Add style instructions based on category
        if category in ['clothing', 'fashion']:
            prompt_parts.append("Add rain effects, bright colors, and bold fonts.")
            prompt_parts.append("Include icons for clothes, monsoon, and shopping.")
        elif category in ['food', 'restaurant']:
            prompt_parts.append("Show delicious food, appetizing colors, steam rising from food.")
            prompt_parts.append("Include food icons and tableware.")
        elif category in ['electronics', 'tech']:
            prompt_parts.append("Add digital elements, circuit patterns, and futuristic style.")
            prompt_parts.append("Include tech gadget icons.")
        
        prompt_parts.append("Format as a square (1:1), suitable for Instagram.")
        prompt_parts.append("High quality, professional advertising poster, bold fonts, eye-catching design.")
        
        # Join the prompt parts
        prompt = "\n".join(prompt_parts)
        
        # For demonstration, create a simple image with text
        # In production, this would be replaced with Stable Diffusion or similar
        img = Image.new('RGB', (512, 512), color=(52, 152, 219))
        
        # Save the image to a temporary file
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        img_path = os.path.join(temp_dir, f"offer_image_{timestamp}.png")
        img.save(img_path)
        
        return img_path, prompt
    except Exception as e:
        # Create an error image
        img = Image.new('RGB', (512, 512), color=(231, 76, 60))
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        img_path = os.path.join(temp_dir, f"offer_image_error_{timestamp}.png")
        img.save(img_path)
        return img_path, f"Error generating image: {str(e)}"

def predict(input_text, temperature, model_path, generate_image):
    """
    Generate a structured offer from input text and optionally an image
    """
    if not os.path.exists(model_path):
        return (f"Error: Model not found at {model_path}. Please train the model first or specify the correct path.", None, None)
    
    try:
        # Load the model
        generator = load_model(model_path)
        
        # Generate offer
        result = generator.generate_offer(input_text, temperature=temperature)
        
        # Format for better display
        formatted_result = format_json(result)
        
        # Generate image if requested
        image_path = None
        prompt_used = None
        if generate_image:
            image_path, prompt_used = generate_image_from_json(formatted_result)
        
        return formatted_result, image_path, prompt_used
    except Exception as e:
        return f"Error generating offer: {str(e)}", None, None

def create_ui():
    """
    Create and configure the Gradio interface
    """
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 DailyFlash Offer Generator")
        gr.Markdown("""
        This tool converts raw promotional text into structured JSON offers and promotional images.
        
        ## Instructions:
        1. Enter promotional text (e.g., "50% off on all electronics at TechMart until Dec 25")
        2. Adjust temperature if needed (higher = more creative, lower = more focused)
        3. Choose whether to generate an image
        4. Click 'Generate' to create a structured offer and optional image
        """)
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Promotional Text",
                    placeholder="Enter promotional text here...",
                    lines=3
                )
                
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.7, 
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness (0.1=focused, 1.0=creative)"
                    )
                    
                    model_path = gr.Textbox(
                        label="Model Path", 
                        value=DEFAULT_MODEL_PATH,
                        info="Path to trained model directory"
                    )
                
                generate_image = gr.Checkbox(
                    label="Generate Promotional Image",
                    value=True,
                    info="Create an image based on the generated offer"
                )
                
                generate_btn = gr.Button("✨ Generate Offer", variant="primary")
                
                # Example inputs
                gr.Examples(
                    examples=[
                        ["Buy 2 get 1 free on shirts at ABC Store, Lucknow. Call: +919876543210"],
                        ["50% off on all electronics at TechMart until Dec 25. Shop now at www.techmart.com"],
                        ["Flash sale: All pizzas at 199 only! Visit Pizza Palace, Delhi before 8 PM today"],
                        ["Monsoon Madness! Up to 70% off on all furniture at HomeDecor. Visit our stores in Mumbai and Pune."]
                    ],
                    inputs=[input_text],
                )
            
            with gr.Column():
                output = gr.JSON(label="Generated Offer")
                image_output = gr.Image(label="Promotional Image", visible=False)
                prompt_output = gr.Textbox(label="Image Prompt", lines=8, visible=False)
        
        # Function to handle UI updates based on checkbox
        def update_image_visibility(generate_image):
            return {
                image_output: gr.update(visible=generate_image),
                prompt_output: gr.update(visible=generate_image)
            }
        
        # Set up event handlers
        generate_btn.click(
            fn=predict,
            inputs=[input_text, temperature, model_path, generate_image],
            outputs=[output, image_output, prompt_output]
        )
        
        generate_image.change(
            fn=update_image_visibility,
            inputs=[generate_image],
            outputs=[image_output, prompt_output]
        )
        
    return app

if __name__ == "__main__":
    # Create and launch the UI
    app = create_ui()
    app.launch(share=True, inbrowser=True)
