"""
Test script for loading and testing real HuggingFace transformer models.
This script validates our enhanced transformer implementation with actual models.
"""

import torch
import json
import argparse
import os
import time
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import AutoModel, AutoTokenizer, AutoConfig
import psutil
import gc


class ModelTester:
    """Test suite for HuggingFace transformer models."""

    def __init__(self, model_name: str, max_seq_len: int = 512):
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.model = None
        self.tokenizer = None
        self.config = None

    def load_model(self) -> Dict[str, Any]:
        """Load a HuggingFace model and return its configuration."""
        print(f"Loading model: {self.model_name}")

        # Load tokenizer and config
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)

        # Load model
        start_time = time.time()
        self.model = AutoModel.from_pretrained(self.model_name)
        load_time = time.time() - start_time

        print(f"Model loaded in {load_time:.2f} seconds")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        return {
            "model_name": self.model_name,
            "load_time": load_time,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "config": self.config.to_dict(),
        }

    def test_inference(self, test_texts: List[str]) -> Dict[str, Any]:
        """Test model inference on sample texts."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        results = []
        total_time = 0

        for i, text in enumerate(test_texts):
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_seq_len,
                truncation=True,
                padding=True,
            )

            # Run inference
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
            inference_time = time.time() - start_time
            total_time += inference_time

            # Get output shape
            output_shape = outputs.last_hidden_state.shape

            results.append(
                {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "input_length": inputs["input_ids"].shape[1],
                    "output_shape": list(output_shape),
                    "inference_time": inference_time,
                    "memory_usage": psutil.Process().memory_info().rss
                    / 1024
                    / 1024,  # MB
                }
            )

            print(f"Text {i+1}: {inference_time:.3f}s, shape: {output_shape}")

        return {
            "total_inference_time": total_time,
            "average_inference_time": total_time / len(test_texts),
            "results": results,
        }

    def extract_model_structure(self) -> Dict[str, Any]:
        """Extract the model structure for our Lean implementation."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Get model dimensions
        d_model = getattr(self.config, "hidden_size", 768)
        num_heads = getattr(self.config, "num_attention_heads", 12)
        num_layers = getattr(self.config, "num_hidden_layers", 12)
        vocab_size = getattr(self.config, "vocab_size", 30000)
        max_position_embeddings = getattr(self.config, "max_position_embeddings", 512)

        # Extract embeddings
        token_embeddings = []
        positional_embeddings = []

        if hasattr(self.model, "embeddings"):
            if hasattr(self.model.embeddings, "word_embeddings"):
                token_embeddings = (
                    self.model.embeddings.word_embeddings.weight.detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
            if hasattr(self.model.embeddings, "position_embeddings"):
                positional_embeddings = (
                    self.model.embeddings.position_embeddings.weight.detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

        return {
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "vocab_size": vocab_size,
            "max_seq_len": max_position_embeddings,
            "token_embeddings_shape": (
                [len(token_embeddings), len(token_embeddings[0])]
                if token_embeddings
                else [0, 0]
            ),
            "positional_embeddings_shape": (
                [len(positional_embeddings), len(positional_embeddings[0])]
                if positional_embeddings
                else [0, 0]
            ),
        }

    def memory_analysis(self) -> Dict[str, Any]:
        """Analyze memory usage of the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Get memory usage
        process = psutil.Process()
        memory_info = process.memory_info()

        # Calculate model size
        param_size = 0
        buffer_size = 0

        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size_mb = (param_size + buffer_size) / 1024 / 1024

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "model_size_mb": model_size_mb,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
        }

    def cleanup(self):
        """Clean up model and free memory."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def test_small_models():
    """Test with small, manageable models."""
    small_models = [
        "distilbert-base-uncased",  # ~66M parameters
        "microsoft/DialoGPT-small",  # ~117M parameters
        "gpt2",  # ~124M parameters
    ]

    test_texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "This is a test of the transformer model capabilities.",
        "Natural language processing has made significant progress.",
    ]

    results = {}

    for model_name in small_models:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"{'='*60}")

        try:
            tester = ModelTester(model_name)

            # Load model
            load_info = tester.load_model()

            # Test inference
            inference_results = tester.test_inference(test_texts)

            # Extract structure
            structure_info = tester.extract_model_structure()

            # Memory analysis
            memory_info = tester.memory_analysis()

            results[model_name] = {
                "load_info": load_info,
                "inference_results": inference_results,
                "structure_info": structure_info,
                "memory_info": memory_info,
                "status": "success",
            }

            # Cleanup
            tester.cleanup()

        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            results[model_name] = {"status": "error", "error": str(e)}

    return results


def test_medium_models():
    """Test with medium-sized models."""
    medium_models = [
        "bert-base-uncased",  # ~110M parameters
        "roberta-base",  # ~125M parameters
        "microsoft/DialoGPT-medium",  # ~345M parameters
    ]

    test_texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
    ]

    results = {}

    for model_name in medium_models:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"{'='*60}")

        try:
            tester = ModelTester(model_name, max_seq_len=256)  # Reduced for memory

            # Load model
            load_info = tester.load_model()

            # Test inference
            inference_results = tester.test_inference(test_texts)

            # Extract structure
            structure_info = tester.extract_model_structure()

            # Memory analysis
            memory_info = tester.memory_analysis()

            results[model_name] = {
                "load_info": load_info,
                "inference_results": inference_results,
                "structure_info": structure_info,
                "memory_info": memory_info,
                "status": "success",
            }

            # Cleanup
            tester.cleanup()

        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            results[model_name] = {"status": "error", "error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Test HuggingFace transformer models")
    parser.add_argument(
        "--model_size",
        choices=["small", "medium", "all"],
        default="small",
        help="Size of models to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_results.json",
        help="Output file for test results",
    )

    args = parser.parse_args()

    print("Starting HuggingFace model testing...")
    print(f"Testing {args.model_size} models")

    all_results = {}

    if args.model_size in ["small", "all"]:
        print("\nTesting small models...")
        small_results = test_small_models()
        all_results["small_models"] = small_results

    if args.model_size in ["medium", "all"]:
        print("\nTesting medium models...")
        medium_results = test_medium_models()
        all_results["medium_models"] = medium_results

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nTest results saved to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for category, models in all_results.items():
        print(f"\n{category.upper()}:")
        for model_name, result in models.items():
            if result["status"] == "success":
                load_time = result["load_info"]["load_time"]
                param_count = result["load_info"]["parameter_count"]
                avg_inference = result["inference_results"]["average_inference_time"]
                memory_mb = result["memory_info"]["rss_mb"]
                print(
                    f"  {model_name}: {param_count:,} params, {load_time:.2f}s load, {avg_inference:.3f}s avg inference, {memory_mb:.1f}MB RAM"
                )
            else:
                print(f"  {model_name}: ERROR - {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
