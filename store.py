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
        last_logits = out.logits[:, -1, :].contiguous()
        tensors_to_transfer.append(last_logits)
        shapes.append(list(last_logits.shape))
        dtypes.append(str(last_logits.dtype))

        # Flatten all tensors into a single contiguous buffer
        flat_tensors = [t.contiguous().view(-1) for t in tensors_to_transfer]
        combined = torch.cat(flat_tensors)

        # Create RDMA buffer from the combined tensor
        buffer_bytes = combined.view(torch.uint8)
        rdma_buffer = RDMABuffer(buffer_bytes)

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
        # Allocate local buffer for RDMA transfer
        shapes = metadata["shapes"]
        dtypes = metadata["dtypes"]
        num_layers = metadata["num_layers"]

        # Calculate total size needed
        total_elements = sum(torch.prod(torch.tensor(shape)).item() for shape in shapes)

        # Allocate buffer and do RDMA read
        local_buffer = torch.empty(total_elements, device=self.device)
        buffer_bytes = local_buffer.view(torch.uint8)

        # Zero-copy RDMA transfer
        await rdma_buffer.read_into(buffer_bytes)

        # Reconstruct tensors from the flat buffer
        offset = 0
        tensors = []
        for shape, dtype_str in zip(shapes, dtypes):
            dtype = getattr(torch, dtype_str.split(".")[-1])
            num_elements = torch.prod(torch.tensor(shape)).item()
            tensor = (
                local_buffer[offset : offset + num_elements].view(dtype).reshape(shape)
            )
            tensors.append(tensor)
            offset += num_elements

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
