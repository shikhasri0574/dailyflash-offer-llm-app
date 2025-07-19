"""
Gradio UI for the DailyFlash Offer Generator.
This script creates a simple web interface for testing the offer generation model.
"""

import os
import json
import gradio as gr
from generate_offer import load_model
import re

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

def predict(input_text, temperature, model_path):
    """
    Generate a structured offer from input text
    """
    if not os.path.exists(model_path):
        return (f"Error: Model not found at {model_path}. Please train the model first or specify the correct path.")
    
    try:
        # Load the model
        generator = load_model(model_path)
        
        # Generate offer
        result = generator.generate_offer(input_text, temperature=temperature)
        
        # Format for better display
        formatted_result = format_json(result)
        
        return formatted_result
    except Exception as e:
        return f"Error generating offer: {str(e)}"

def create_ui():
    """
    Create and configure the Gradio interface
    """
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 DailyFlash Offer Generator")
        gr.Markdown("""
        This tool converts raw promotional text into structured JSON offers.
        
        ## Instructions:
        1. Enter promotional text (e.g., "50% off on all electronics at TechMart until Dec 25")
        2. Adjust temperature if needed (higher = more creative, lower = more focused)
        3. Click 'Generate' to create a structured offer
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
                
        # Set up event handler
        generate_btn.click(
            fn=predict,
            inputs=[input_text, temperature, model_path],
            outputs=output
        )
        
    return app

if __name__ == "__main__":
    # Create and launch the UI
    app = create_ui()
    app.launch(share=True, inbrowser=True)
