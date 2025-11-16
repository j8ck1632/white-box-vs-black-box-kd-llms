"""
DistillationStudent Model Class

This module contains the DistillationStudent class that wraps TinyLlama
and adds trainable adapter layers for knowledge distillation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class DistillationStudent(nn.Module):
    """
    A PyTorch model that wraps TinyLlama and adds adapter layers for knowledge distillation.
    
    This model includes:
    - The base TinyLlama model (trainable)
    - A hidden state projector to align student hidden states with teacher dimensions
      (2048 -> 4096)
    """
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
        """
        Initialize the DistillationStudent model.
        
        Args:
            model_name: HuggingFace model identifier for the student model
            device: Optional device to load model on. If None, uses "cuda" if available, else "cpu"
        """
        super().__init__()
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        bf16_supported = False
        if device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            bf16_supported = capability[0] >= 8  # Ampere or newer

        if device == "cuda" and torch.cuda.is_available():
            preferred_dtype = torch.bfloat16 if bf16_supported else torch.float16
            self.student_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=preferred_dtype,
                device_map="auto",
            )
        else:
            self.student_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            )
            self.student_model = self.student_model.to(device)
        
        # Get the hidden dimension from the model config
        self.d_student = self.student_model.config.hidden_size  # Should be 2048
        self.d_teacher = 4096  # Teacher (Mistral-7B) hidden dimension
        self._eager_attention_enabled = False
        
        # Trainable adapter layer to project student hidden states to teacher dimension
        # This is crucial for aligning hidden states between student and teacher
        self.hidden_state_projector = nn.Linear(
            in_features=self.d_student,
            out_features=self.d_teacher
        )
        
        # Initialize the projector with small random weights
        nn.init.xavier_uniform_(self.hidden_state_projector.weight)
        nn.init.zeros_(self.hidden_state_projector.bias)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_hidden_states: bool = False,
        return_attention: bool = False,
        output_attentions: bool = False
    ):
        """
        Forward pass through the student model.
        
        Args:
            input_ids: Tokenized input sequence [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden_states: If True, return the final layer hidden states
            return_attention: If True, return the final layer attention maps
            output_attentions: If True, the model will output attentions
            
        Returns:
            Dictionary containing:
            - logits: Model output logits [batch_size, seq_len, vocab_size]
            - hidden_state: Final layer hidden states [batch_size, seq_len, d_student] (if requested)
            - projected_hidden_state: Projected hidden states [batch_size, seq_len, d_teacher] (if requested)
            - attention_map: Final layer attention maps [batch_size, num_heads, seq_len, seq_len] (if requested)
        """
        need_attention = return_attention or output_attentions
        if need_attention and not self._eager_attention_enabled:
            set_attn_impl = getattr(self.student_model, "set_attn_implementation", None)
            if callable(set_attn_impl):
                try:
                    set_attn_impl("eager")
                    self._eager_attention_enabled = True
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"WARNING: Failed to set eager attention: {exc}")

        # Forward pass through the base model
        outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=need_attention
        )
        
        # Extract logits (always returned)
        logits = outputs.logits
        
        result = {"logits": logits}
        
        # Extract and project hidden states if requested
        if return_hidden_states:
            # Get the final layer hidden states (last element of hidden_states tuple)
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, d_student]
            projector_dtype = self.hidden_state_projector.weight.dtype
            if hidden_states.dtype != projector_dtype:
                hidden_states = hidden_states.to(projector_dtype)
            
            # Project to teacher dimension
            projected_hidden_state = self.hidden_state_projector(hidden_states)
            
            result["hidden_state"] = hidden_states
            result["projected_hidden_state"] = projected_hidden_state
        
        # Extract attention maps if requested
        if return_attention and outputs.attentions is not None:
            # Get the final layer attention maps (last element of attentions tuple)
            attention_map = outputs.attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
            result["attention_map"] = attention_map
        
        return result
    
    def get_tokenizer(self):
        """
        Get the tokenizer associated with the student model.
        
        Returns:
            AutoTokenizer instance
        """
        return AutoTokenizer.from_pretrained(
            self.student_model.config.name_or_path
        )

