from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn.functional as F

import lm_eval
from lm_eval.base import BaseLM
from lm_eval.evaluator import evaluate, make_table
from megatron.core.models.gpt import GPTModel
from pretrain_gpt import model_provider
from megatron.training.checkpointing import load_checkpoint
from megatron.training import get_tokenizer
from megatron.training.initialize import initialize_megatron
from megatron.core.pipeline_parallel.p2p_communication import recv_forward, send_forward
from megatron.core.pipeline_parallel.schedules import get_tensor_shapes
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from megatron.training import get_args, get_model
from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
from megatron.core.tensor_parallel.utils import VocabUtility
import megatron.core.tensor_parallel as TP
from megatron.training.arguments import parse_args
from megatron.core import mpu
from megatron.core import parallel_state

from megatron.core.utils import get_model_config, get_model_type

class GPTModelWrapped(BaseLM):

    def __init__(self, model: GPTModel, tokenizer: get_tokenizer, device="cuda", temperature=0.8, top_k=200,
                 max_gen_tokens=128, batch_size=1, eot_token=2, max_length=1024):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_gen_tokens = max_gen_tokens
        self._batch_size = batch_size
        self._eot_token = eot_token
        self._temperature = temperature
        self._top_k = top_k
        self._max_length = max_length
        self.config = core_transformer_config_from_args(args)


    @property
    def eot_token_id(self):
        return self._eot_token

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_tokens

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self._tokenizer.tokenize(string)

    def tok_decode(self, tokens: Iterable[int]):
        return self._tokenizer.detokenize(tokens.tolist())

    def _model_generate(self, context, max_length, eos_token_id):
        return self._model.generate(context, max_length, temperature=self._temperature, top_k=self._top_k, eos_token=eos_token_id)

    def _pad_input_and_attn_mask(self, inps, max_length):
        
        # Pad the input to max_length
        padding_length = max_length - inps.size(-1)
        if padding_length > 0:
            # inps = F.pad(inps, (padding_length, 0), value=-1)
             F.pad(inps, (0, 0, 0, padding_length), value = 0)
        return inps

    def _model_call(self, inps):
        args = get_args()
        tokenizer = self._tokenizer

        inps_padded = self._pad_input_and_attn_mask(inps, self.max_length)
        # Get the masks and postition ids.
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            inps,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        output = []
        a_output = model(inps_padded, position_ids.unsqueeze(dim=0), attention_mask)
        output.append(a_output)

        return torch.cat(output, 0)[:len(inps)] #if mpu.is_pipeline_last_stage() else None


if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = 'out'  # ignored if init_from is not 'resume'
    max_new_tokens = 500  # number of tokens generated in each sample
    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' #if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
    tasks = ["wikitext", "lambada_openai", "hellaswag"]  # examples: --tasks='["lambada_openai"]'
    limit = 100
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    initialize_megatron()

    # config = core_transformer_config_from_args(args)

    args = get_args()
    print(args)
    # Set up model and load checkpoint.
    model = get_model(model_provider, wrap_with_ddp=False, parallel_output=False)
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    model.eval()
    model.to(device)

    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    with torch.no_grad():
        # with ctx:
        results = evaluate(
            lm=GPTModelWrapped(model, get_tokenizer(), device=device, max_gen_tokens=max_new_tokens, temperature=temperature, top_k=top_k, max_length=args.seq_length),
            task_dict=lm_eval.tasks.get_task_dict(tasks),  limit=limit
        )
        print(make_table(results))
