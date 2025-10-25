from __future__ import annotations
import asyncio
import torch
from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import uuid


class PrefillWorker(Actor):
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        )
        self.rdma_buffers = {}
        self.cache_metadata = {}

    @endpoint
    async def prefill(self, key: str, prompt: str) -> tuple[RDMABuffer, dict]:
        enc = self.tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        cache = DynamicCache(config=self.model.config)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)

        tensors_to_transfer = []
        shapes = []
        dtypes = []

        # Add all key-value pairs from the cache
        for layer_idx in range(len(out.past_key_values)):
            key_tensor, value_tensor = out.past_key_values[layer_idx]
            tensors_to_transfer.extend([key_tensor, value_tensor])
            shapes.extend([list(key_tensor.shape), list(value_tensor.shape)])
            dtypes.extend([str(key_tensor.dtype), str(value_tensor.dtype)])

        # Add the logits (last token only)
        last_logits = out.logits[:, -1, :]
        tensors_to_transfer.append(last_logits)
        shapes.append(list(last_logits.shape))
        dtypes.append(str(last_logits.dtype))

        # Calculate total bytes needed
        total_bytes = 0
        for tensor in tensors_to_transfer:
            total_bytes += tensor.numel() * tensor.element_size()

        # Allocate single contiguous buffer
        flat_buffer = torch.empty(total_bytes, dtype=torch.uint8, device=self.device)

        # Copy each tensor directly into the buffer (this is the only copy - unavoidable)
        byte_offset = 0
        for tensor in tensors_to_transfer:
            tensor_bytes = tensor.numel() * tensor.element_size()
            flat_buffer[byte_offset : byte_offset + tensor_bytes].copy_(
                tensor.view(-1).view(torch.uint8)
            )
            byte_offset += tensor_bytes

        # Create RDMA buffer from the single flat buffer - no copy, just a reference
        rdma_buffer = RDMABuffer(flat_buffer)

        # Store buffer and metadata
        self.rdma_buffers[key] = rdma_buffer
        metadata = {
            "shapes": shapes,
            "dtypes": dtypes,
            "num_layers": len(out.past_key_values),
        }
        self.cache_metadata[key] = metadata

        return rdma_buffer, metadata


class DecodeWorker(Actor):
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        )

    @endpoint
    async def decode(
        self, rdma_buffer: RDMABuffer, metadata: dict, max_new_tokens: int = 16
    ) -> str:
        shapes = metadata["shapes"]
        dtypes = metadata["dtypes"]
        num_layers = metadata["num_layers"]

        # Calculate total size in bytes for a single flat allocation
        total_bytes = 0
        element_sizes = []
        for shape, dtype_str in zip(shapes, dtypes):
            dtype = getattr(torch, dtype_str.split(".")[-1])
            num_elements = torch.prod(torch.tensor(shape)).item()
            element_size = torch.tensor([], dtype=dtype).element_size()
            total_bytes += num_elements * element_size
            element_sizes.append((num_elements, element_size, dtype))

        # Allocate one contiguous buffer as bytes
        flat_buffer = torch.empty(total_bytes, dtype=torch.uint8, device=self.device)

        # Zero-copy RDMA transfer directly into the single flat buffer
        await rdma_buffer.read_into(flat_buffer)

        # Create views (not copies) into the flat buffer for each tensor
        byte_offset = 0
        tensors = []
        for shape, (num_elements, element_size, dtype) in zip(shapes, element_sizes):
            # View into the flat buffer at the correct byte offset
            tensor = (
                flat_buffer[byte_offset : byte_offset + num_elements * element_size]
                .view(dtype)
                .reshape(shape)
            )
            tensors.append(tensor)
            byte_offset += num_elements * element_size

        # Reconstruct past_key_values (tuple of tuples)
        past_key_values = []
        for i in range(num_layers):
            key_tensor = tensors[i * 2]
            value_tensor = tensors[i * 2 + 1]
            past_key_values.append((key_tensor, value_tensor))
        past_key_values = tuple(past_key_values)

        # Get the logits (last tensor)
        next_token_logits = tensors[-1]

        # Generate tokens
        generated_tokens = []
        for i in range(max_new_tokens):
            # Sample next token
            probs = torch.softmax(next_token_logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token.item())

            # Forward pass with cache
            with torch.no_grad():
                outputs = self.model(
                    next_token, past_key_values=past_key_values, use_cache=True
                )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        return self.tok.decode(generated_tokens, skip_special_tokens=True)


async def generate(prompt, prefill, decode, max_new_tokens):
    # Get RDMA buffer reference and metadata from prefill
    rdma_buffer, metadata = await prefill.prefill.call_one(str(uuid.uuid4()), prompt)

    # Decode uses RDMA to get the cache
    text = await decode.decode.call_one(
        rdma_buffer, metadata, max_new_tokens=max_new_tokens
    )

    return text


# -----------------------------
# Driver
# -----------------------------
async def main():
    prefill_proc = this_host().spawn_procs(per_host={"gpus": 1})
    decode_proc = this_host().spawn_procs(per_host={"gpus": 1})

    prefill = prefill_proc.spawn("prefill", PrefillWorker)
    decode = decode_proc.spawn("decode", DecodeWorker)

    prompt = "Hello there,"
    out = await generate(prompt, prefill, decode, max_new_tokens=32)

    print("=== GENERATED ===")
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
