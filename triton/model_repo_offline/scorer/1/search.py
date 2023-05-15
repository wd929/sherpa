import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import k2
from typing import List, Union

def forward_joiner(cur_encoder_out, decoder_out):
    in_joiner_tensor_0 = pb_utils.Tensor.from_dlpack("encoder_out", to_dlpack(cur_encoder_out))
    in_joiner_tensor_1 = pb_utils.Tensor.from_dlpack("decoder_out", to_dlpack(decoder_out.squeeze(1)))

    inference_request = pb_utils.InferenceRequest(
        model_name='joiner',
        requested_output_names=['logit'],
        inputs=[in_joiner_tensor_0, in_joiner_tensor_1])
    inference_response = inference_request.exec()
    if inference_response.has_error():
        raise pb_utils.TritonModelException(inference_response.error().message())
    else:
        # Extract the output tensors from the inference response.
        logits = pb_utils.get_output_tensor_by_name(inference_response,
                                                        'logit')
        logits = torch.utils.dlpack.from_dlpack(logits.to_dlpack()).cpu()
        assert len(logits.shape) == 2, logits.shape
        return logits

def forward_decoder(hyps, context_size):
    decoder_input = [h[-context_size:] for h in hyps]

    decoder_input = np.asarray(decoder_input,dtype=np.int64)

    in_decoder_input_tensor = pb_utils.Tensor("y", decoder_input)

    inference_request = pb_utils.InferenceRequest(
        model_name='decoder',
        requested_output_names=['decoder_out'],
        inputs=[in_decoder_input_tensor])

    inference_response = inference_request.exec()
    if inference_response.has_error():
        raise pb_utils.TritonModelException(inference_response.error().message())
    else:
        # Extract the output tensors from the inference response.
        decoder_out = pb_utils.get_output_tensor_by_name(inference_response,
                                                        'decoder_out')
        decoder_out = from_dlpack(decoder_out.to_dlpack())
        return decoder_out


def greedy_search(encoder_out, encoder_out_lens, context_size, unk_id, blank_id):
    
    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False
    )

    pack_batch_size_list = packed_encoder_out.batch_sizes.tolist()
            
    hyps = [[blank_id] * context_size for _ in range(encoder_out.shape[0])]
    decoder_out = forward_decoder(hyps, context_size)

    offset = 0
    for batch_size in pack_batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = packed_encoder_out.data[start:end]

        offset = end
    
        decoder_out = decoder_out[:batch_size]

        logits = forward_joiner(current_encoder_out, decoder_out)

        assert logits.ndim == 2, logits.shape
        y = logits.argmax(dim=1).tolist()
        
        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(v)
                emitted = True
        if emitted:
            decoder_out = forward_decoder(hyps[:batch_size], context_size)


    sorted_ans = [h[context_size:] for h in hyps]

    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(encoder_out.shape[0]):
        ans.append(sorted_ans[unsorted_indices[i]])

    return ans

# From k2 utils.py
def get_texts(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> Union[List[List[int]], k2.RaggedTensor]:
    """Extract the texts (as word IDs) from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      return_ragged:
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        # TODO: change arcs.shape() to arcs.shape
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)

        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2
    if return_ragged:
        return aux_labels
    else:
        return aux_labels.tolist()

def fast_beam_search(encoder_out, encoder_out_lens, context_size, vocab_size, beam, max_contexts, max_states, decoding_graph):
    B, T, C = encoder_out.shape
  
    config = k2.RnntDecodingConfig(
        vocab_size=vocab_size,
        decoder_history_len=context_size,
        beam=beam,
        max_contexts=max_contexts,
        max_states=max_states,
    )
    individual_streams = []
    for i in range(B):
        individual_streams.append(k2.RnntDecodingStream(decoding_graph))
    decoding_streams = k2.RnntDecodingStreams(individual_streams, config)
  
    for t in range(T):
        shape, contexts = decoding_streams.get_contexts()
        contexts = contexts.to(torch.int64)
  
        decoder_out = forward_decoder(contexts)
  
        cur_encoder_out = torch.index_select(
            encoder_out[:, t:t + 1, :], 0, shape.row_ids(1).to(torch.int64)
        )
  
        logits = forward_joiner(cur_encoder_out.squeeze(1),
            decoder_out)
  
        logits = logits.squeeze(1).squeeze(1).float()
        log_probs = (logits / self.temperature).log_softmax(dim=-1)
        decoding_streams.advance(log_probs)
    decoding_streams.terminate_and_flush_to_streams()
    lattice = decoding_streams.format_output(encoder_out_lens.tolist())
  
    best_path = k2.shortest_path(lattice, use_double_scores=True)
    hyps_list = get_texts(best_path)

    return hyps_list
