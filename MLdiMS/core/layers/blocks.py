#initial version
import functools
from pprint import pprint
from typing import Any, Callable, Dict, Literal, Tuple, Mapping, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .baseops import (
        Dense,
        RMSNorm,
        DenseGeneral,
        Dropout,
        relu,
        silu,
        AttnOutput,
        SinusoidalPositionalEncoding,
        LearnedPositionalEncoding,
        RotaryPositionalEncoding,
        dot_product_attention,
        softmax,
        Embed,
        Linear,
        gelu,
        normalize_attention,
        ShardMixIn
    )

from core.layers.kvcache import kvcache


##MLPBlock building block of the transformer
#linear layer scaling up the hidden dimensionality , non-linearity 
#and a linear layer scaling down the hidden dimensionality

class MLPBlockInput(nn.Module):
    features: int
    data_type : jnp.dtype
    use_norm : bool = True
    use_bias : bool = False
    kernel_init: Callable = nn.initializers.lecun_normal()


    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_norm:
            x = RMSNorm(dtype=self.dtype, name="MLPPreNorm", epsilon=self.epsilon)(x)
        x = Dense(
            features=self.features,
            kernel_init = self.kernel_init,
            use_bias = self.use_bias,
            dtype=self.data_type,
            name="MLPDense"
        )(x)
        #x = gelu()(x)
        x = nn.gelu(x)
        return x

class MLPBlockOutput(nn.Module):
    features: int
    data_type : jnp.dtype
    use_bias : bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = Dense(
            features=self.features,
            kernel_init = self.kernel_init,
            use_bias = self.use_bias,
            dtype=self.data_type,
            name="MLPDense"
        )(x)
        return x

class MLPBlock(nn.Module):
     hidden_dim: int
     embedding_dim: int
     data_type: jnp.dtype
     weight_dtype: jnp.dtype
     use_norm: bool = False
     use_bias: bool = True
     in_shard_axes: Optional[Tuple[str,...]] = None
     out_shard_axes: Optional[Tuple[str,...]] = None
     kernel_init : Callable = nn.initializers.xavier_uniform()

     def setup(self):
         super().setup()

     @nn.compact
     def __call__(self, x: jax.Array) -> jax.Array:
         input_features = x[-1]

         if self.use_norm:
             x = RMSNorm(data_type=self.data_type,name="MLPPrenorm")(x)
         x = DenseGeneral(
             features=self.hidden_dim,
             dtype=self.data_type,
             param_dtype=self.weight_dtype,
             kernel_init=self.kernel_init,
             name="MLPInp",
             shard_axes={"MLPInp": self.in_shard_axes},
             )(x)
         x = nn.gelu(x)
         x = DenseGeneral(
             features=self.embedding_dim,
             dtype=self.data_type,
             param_dtype=self.weight_dtype,
             kernel_init=self.kernel_init,
             name="MLPOut",
             shard_axes={"MLPOut": self.out_shard_axes},
             )(x)
         return x

class QKVProjection(nn.Module):
    
    use_bias: bool
    head_dim: int
    num_heads: int
    data_type: jnp.dtype
    kernel_init : Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros
    normalize_qk : bool = False

    def setup(self):
        super().setup()


    @nn.compact
    def __call__(self,x: jax.Array) -> tuple [jax.Array,jax.Array,jax.Array]:
        q = DenseGeneral(
                (self.num_heads,self.head_dim),
                kernel_init = self.kernel_init,
                bias_init = self.bias_init,
                dtype= self.data_type,
                use_bias = self.use_bias,
                name="QProj",
        )(x)
        k = DenseGeneral(
                (self.num_heads,self.head_dim),
                kernel_init = self.kernel_init,
                bias_init = self.bias_init,
                dtype= self.data_type,
                use_bias = self.use_bias,
                name="KProj",
        )(x)
        v = DenseGeneral(
                (self.num_heads,self.head_dim),
                kernel_init = self.kernel_init,
                bias_init = self.bias_init,
                dtype= self.data_type,
                use_bias = self.use_bias,
                name="VProj",
        )(x)
        if self.normalize_qk:
            q = RMSNorm(
                dtype=self.data_type,
                name="Qnorm",
            )(q)
            k = RMSNorm(
                dtype=self.data_type,
                name="Knorm",
            )(k)
        return q,k,v


class QKVFusedAttnInput(nn.Module):
    
    head_dim: int
    num_heads: int
    dtype: jnp.dtype
    weight_dtype : jnp.dtype
    kernel_init : Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros
    #shard_axes: Optional[Mapping[str, Tuple[str,...]]] = None 
    shard_axes: Optional[Tuple[str,...]] = None 
    use_bias: bool = False

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self,x: jax.Array) -> tuple [jax.Array,jax.Array,jax.Array]:
        qkv = DenseGeneral(
                (3,self.num_heads,self.head_dim),
                kernel_init = self.kernel_init,
                bias_init = self.bias_init,
                dtype= self.dtype,
                param_dtype =  self.weight_dtype,
                use_bias = self.use_bias,
                name="QKVProj",
                shard_axes={"QKVProj": self.shard_axes},
        )(x)
        return qkv[:,:,0,...], qkv[:,:,1,...], qkv[:,:,2,...]


class AttnInput(nn.Module):
    
    head_dim: int
    num_heads: int
    dtype: jnp.dtype
    weight_dtype : jnp.dtype
    use_bias: bool = False
    kernel_init : Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros
    shard_axes: Optional[Tuple[str,...]] = None 
    #shard_axes: Optional[Mapping[str, Tuple[str,...]]] = None 

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self,x: jax.Array) -> tuple [jax.Array,jax.Array,jax.Array]:
        out = DenseGeneral(
                (self.num_heads,self.head_dim),
                kernel_init = self.kernel_init,
                bias_init = self.bias_init,
                dtype= self.data_type,
                param_dtype = self.weight_dtype,
                use_bias = self.use_bias,
                name="AttnInp",
                shard_axes={"AttnInp": self.shard_axes},
        )(x)
        return out 

class KVAttnInput(nn.Module):
    
    head_dim: int
    num_heads: int
    data_type: jnp.dtype
    use_bias: bool = False
    kernel_init : Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros
    #shard_axes: Optional[Mapping[str, Tuple[str,...]]] = None 
    shard_axes: Optional[Tuple[str,...]] = None 

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self,x: jax.Array) -> tuple [jax.Array,jax.Array,jax.Array]:
        kv = DenseGeneral(
                (self.num_heads,self.head_dim),
                kernel_init = self.kernel_init,
                bias_init = self.bias_init,
                dtype= self.data_type,
                use_bias = self.use_bias,
                name="KVProj",
                shard_axes={"KVProj": self.shard_axes},
        )(x)
        return kv

class AttnOut(nn.Module):
    features: int
    data_type: jnp.dtype
    use_bias: bool = True
    kernel_init : Callable = nn.initializers.lecun_normal()

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        x = DenseGeneral(
            features=self.features,
            axes=(-2,-1),
            data_type=self.data_type,
            use_bias = self.use_bias,
            kernel_init=self.kernel_init,
        )(x)
        return x

class MultiHeadAttention(nn.Module):
     embedding_dim : int  #model dimension 
     num_heads: int
     use_bias_qkv: bool
     attention: Callable
     data_type: jnp.dtype
     mask : jax.Array | None = None
     normalize_qk: bool = False

     def setup(self):
         super().setup()

     @nn.compact
     def __call__(self,x: jax.Array) -> jax.Array:
         features = x.shape[-1]
         #Normalize lyaer
         x = RMSNorm(dtype=self.data_type,name="PreNorm")(x)
         #QKV projection
         attn_dim = self.embedding_dim//self.num_heads
         q,k,v = QKVProjection(
                 head_dim=attn_dim,
                 num_heads = self.num_heads,
                 use_bias = self.use_bias_qkv,
                 normalize_qk = self.normalize_qk,
                 data_type = self.data_type,
                 name="QKVProjection"
         )(x)
         x = dot_product_attention(
             q,k,v,self.mask,
             )(x)
         x = AttnOutput(
             features=features,
             data_type = self.data_type,
             name="AttnOut",
         )(x)
         return x


class GenericAttention(nn.Module):
    """ Attention layer for Generic LLM
    
    attributes:
      num_heads: int
      kv_div_factor : int  (query->kv heads ratio)
      attn_dim: int
      mesh: jax Mesh
      attention_kernel: Callable[,...]
      dtype: activation data type
      weight_dtype : weight data type
      attention_bias: bool
      kernel_init : parameter initializer
      max_seq_len : maximum sequence length
      max_prefill_predict_len : maximum prefill sequence length in append phase
      dropout_rate: float = 0
      shard_axes: Mapping[str, Tuple[str,...]]

    """

    num_heads : int
    kv_div_factor: int
    attn_dim: int
    dtype: jnp.dtype
    weight_dtype: jnp.dtype
    attention_bias: bool
    max_seq_len : int
    rope_min_timescale: int
    rope_max_timescale: int
    dropout_rate: float = 0.0
    qkv_bias: bool = False
    max_prefill_predict_len: int  = -1
    shard_axes : Optional[Mapping[str,Tuple[str,...]]] = None
    kernel_init: Callable = nn.initializers.variance_scaling(1.0,"fan_in","normal")
    attention_kernel: Callable[...,nn.Module] | None = None


    prefill_dim_order: tuple = (1,2,0,3)
    append_dim_order: tuple = (1,2,0,3)
    compute_dim_order: tuple = (0,1,2,3)

    mask_value = -0.7 ** float(np.finfo(np.dtype("float32")).max)

    def setup(self):
        super().setup()
        self._cache = kvcache(self.max_seq_len,
            self.max_prefill_predict_len,
            self.shard_axes["prefill_axis"],
            self.shard_axes["append_axis"],
            self.prefill_dim_order,
            self.append_dim_order,
            self.dtype
        )


    def generate_attention_mask(self, query, key, decoder_segment_ids: jax.Array | None, mode: str) -> jax.Array | None:
    
        mask = None
        if mode == "append":
            mask = decoder_segment_ids[:, None, None, None, :] == 1
        elif decoder_segment_ids is not None:
            mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
            mask = mask[:, None, None, :, :]

        causal_mask = None
        # We enforce causality except for append
        if mode != "append":
            q_seq_len = query.shape[1]
            kv_seq_len = key.shape[1]
            mask_shape = (q_seq_len, kv_seq_len)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            causal_mask = (col_ids <= row_ids)[None, None, None, :, :]

        output_mask = None

        if (mask is not None) and (causal_mask is not None):
            output_mask = jnp.logical_and(mask, causal_mask)
        elif mask is not None:
            output_mask = mask
        elif causal_mask is not None:
            output_mask = causal_mask

        return jnp.where(output_mask, 0.0, self.mask_value) if output_mask is not None else None

    #@functools.partial(jax.jit, static_argnums=(2,3))
    def apply_dot_product_attention(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        decoder_segment_ids: jax.Array,
        mode: str = "prefill"
        ):
        """Apply dot product Attention."""

        q_seq_len = query.shape[1]
        #batch, seqlen, num_heads, attn_dim
        b, t, h, d = query.shape
        #query->kv heads
        #complaining about shape attribute not available for key variable
        #n_kv = key.shape[-2]
        n_kv = self.kv_div_factor
        assert(n_kv == self.kv_div_factor)
        if mode == "training" or self.compute_dim_order == (0, 1, 2, 3):
            query = jnp.reshape(query, (b, t, n_kv, h // n_kv, d))
            attn_weights = jnp.einsum("btkgd,bskd->bkgts", query, key)
        elif self.compute_dim_order == (0, 2, 1, 3):
            query = jnp.transpose(query, axes=self.compute_dim_order)
            key = jax.tree.map(lambda x: jnp.transpose(x, axes=self.compute_dim_order), key)
            query = jnp.reshape(query, (b, n_kv, n // n_kv, t, d))
            attn_weights = jnp.einsum("bkgtd,bksd->bkgts", query, key)

        attn_weights = attn_weights.astype(jnp.float32)
        attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, mode)
        if attn_mask is not None:
            #attn_weights = apply_mask_to_logits(attn_weights, attn_mask)
            attn_weights = jnp.where((attn_mask >= self.mask_value * 0.5), attn_weights, self.mask_value)

        ##softmax
        _max = jnp.max(attn_weights, axis=-1,keepdims=True)
        _exp = jnp.exp(attn_weights-_max)
        _sum = jnp.sum(_exp,axis=-1,keepdims=True)

        _sum = jnp.moveaxis(_sum,-2,1)
        _max = jnp.moveaxis(_max,-2,1)

        _max = jnp.reshape(_max,(_max.shape[0],_max.shape[1],_max.shape[2]*_max.shape[3],1))
        _sum = jnp.reshape(_sum,(_sum.shape[0],_sum.shape[1],_sum.shape[2]*_sum.shape[3],1))

        ##P = SV
        if mode == "training"  or self.compute_dim_order == (0, 1, 2, 3):
            out = jnp.einsum("bkgts,bskd->btkgd", attn_weights, value)
            b, t, n_kv, h, d = out.shape
            result = jnp.reshape(out, (b, t, n_kv * h, d))
        elif self.compute_dim_order == (0, 2, 1, 3):
            value = jax.tree.map(lambda x: jnp.transpose(x, axes=self.compute_dim_order), value)
            out = jnp.einsum("bkgts,bksd->bkgtd", attn_weights, value)
            b, n_kv, g, s, d = out.shape
            result = jnp.reshape(out, (b, n_kv * g, s, d))
            result = jax.numpy.moveaxis(result, (0, 1, 2, 3), self.compute_dim_order)
        return result,_max,_sum

    @nn.compact
    def __call__(
        self,
        x_q: jax.Array,
        x_kv: jax.Array,
        x_position: jax.Array,
        decoder_segment_ids: jax.Array | None = None,
        *,
        mode: Literal["training","Prefill","append"] = "prefill",
        deterministic: bool = False,
        ):

        """
          Attention layer consists of QKV projection + attention layer + Out projection

          inputs: 
           x_q: input queries of shape [batch,seqlen,heads,attn_dim]
           q_kv: input [key,value] of shape [batch, seqlen, heads, attn_dim]
           mode: training or inference
           deterministic: disables dropout if set to True

          returns:
           output  of shape [batch, length, heads, attn_dim]
        """

        ##fused QKV implementation
        q,k,v = QKVFusedAttnInput(
            head_dim=self.attn_dim,
            num_heads=self.num_heads,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init = self.kernel_init,
            use_bias=self.qkv_bias,
            shard_axes=self.shard_axes["QKVProj"],
        )(x_q)

        ## apply ROPE 
        q_rope = RotaryPositionalEncoding(
            embedding_dim = self.attn_dim,
            min_timescale = self.rope_min_timescale,
            max_timescale = self.rope_max_timescale,
            name = "Qrope")(q,x_position)

        k_rope = RotaryPositionalEncoding(
            embedding_dim = self.attn_dim,
            min_timescale = self.rope_min_timescale,
            max_timescale = self.rope_max_timescale,
            name = "Krope")(k,x_position)

        # map sharding axes to activations

        q_rope = nn.with_logical_constraint(q_rope,self.shard_axes["Qrope"])
        k_rope = nn.with_logical_constraint(k_rope,self.shard_axes["Krope"])
        v  = nn.with_logical_constraint(v,self.shard_axes["Vrope"])


        prefill_kvcache, append_kvcache = self._cache(q_rope,k_rope,decoder_segment_ids,mode)
        ##call flashattention 
        if mode == "prefill":
            prefill_out,prefill_max,prefill_sum = self.apply_dot_product_attention(query=q_rope,
                key=prefill_kvcache[0],
                value=prefill_kvcache[1],
                decoder_segment_ids=prefill_kvcache[2],
                mode=mode
            )
            if prefill_out is not None:
                out =  normalize_attention(prefill_out,prefill_max,prefill_sum)
        elif mode == "append":
            append_out,append_max,append_sum = self.apply_dot_product_attention(query=q_rope,
                key=append_kvcache[0],
                value=append_kvcache[1],
                decoder_segment_ids=append_kvcache[2],
                lengths=append_kvcache[3],
                mode=mode,
            )
            if append_out is not None:
                out =  normalize_attention(append_out,append_max,append_sum)
        else:
            raise valueError(f"unsupported mode {mode}")

        out = nn.with_logical_constraint(out,self.shard_axes["AttnOut"])
        out = DenseGeneral(
            features = x_q.shape[-1],
            kernel_init=self.kernel_init,
            name="AttnOut",
            shard_axes={"AttnOut":self.shard_axes["AttnOut"]},
            dtype=self.dtype,
            param_dtype=self.weight_dtype,
            axis=(-2,-1))(out)
        return out

class TransformerBlock(nn.Module):
    embedding_dim: int = 1024
    num_heads: int = 8
    causal_mask: bool = True
    normalize_qk: bool = False
    use_bias_qkv: bool = True
    hidden_dim: int = 4096
    train : bool = True
    data_type: jnp.dtype = jnp.float16
    dropout_rate: jnp.float32 = 0.1

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        #attention layer block
        attn_out  = MultiHeadAttention(
            embedding_dim = self.embedding_dim,
            num_heads = self.num_heads,
            use_bias_qkv = self.use_bias_qkv,
            mask = self.mask,
            data_type= self.data_type,
            )(x)
        attn_out = Dropout(rate=self.dropout_rate, deterministic=not self.train)(attn_out)
        x = x + attn_out
        #MLP block
        x = RMSNorm(data_type=self.data_type,name="MLPPrenorm")(x)
        x = MLPBLockInput(
             features=self.hidden_dim,
             use_norm=False,
             use_bias=False,
             name="MLPInput")(x)
        x = MLPBLockOutput(
             features=self.embedding_dim,
             use_bias=True,
             name="MLPOutput")(x)
        mlp_out = Dropout(
            rate=self.dropout_rate,
            deterministic = not self.train,
        )(mlp_out)
        x = x + mlp_out
        return x

class OutputLayer(nn.Module):
    num_outputs: int
    data_type: jnp.dtype = jnp.float16


    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:

        x = RMSNorm(
            dtype=self.data_type,
            )(x)
        x = Dense(
            features=self.num_ouputs,
            dtype=self.data_type,
            )(x)
        return x

class PositionalEncoding(nn.Module):
  embedding_dim: int
  shard_axis_name: str
  base_exponent: int = 10000

  def __call__(
    self,
    x: jax.Array,
    position: jax.Array,
    ) -> jax.Array:

    seq_len, num_features = x.shape[-2:]
    tp_size = jax.lax.psum(1, self.shard_axis_name)
    num_timescales = self.embedding_dim // 2 // tp_size
    log_timescale_increment = jnp.log(float(self.base_exponent)) / jnp.maximum(
        jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1
    )
    if position is None:
        position = jnp.arange(0, seq_len, dtype=jnp.float32)
        #expand axis in batch dimension
        position = jnp.expand_dims(position,0)
    inv_timescales = jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
    #expand dimension in embedding dimension
    position = position[:, :, jnp.newaxis]
    inv_timescales = inv_timescales[jnp.newaxis, jnp.newaxis, :]
    scaled_time = position * inv_timescales
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
    # signal = jnp.pad(signal, [[0, jnp.mod(self.embedding_dims, 2)]])
    position_embedding = signal.astype(jnp.float32)
    pos_emb = jnp.expand_dims(position_embedding, axis=range(x.ndim - 2))
    return x + position_embedding

class OldPositionalEncoding(nn.Module):
    embedding_dim:int
    shard_axis_name: str
    dtype: jnp.dtype = jnp.float32
    encoding_type: str = "learned"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        #FIXME use argument
        tp_size = jax.lax.psum(1, self.shard_axis_name)
        tp_index = jax.lax.axis_index(self.shard_axis_name)
        seq_len, num_feats = x.shape[-2:]
        #print(num_feats)
        if self.encoding_type == "learned":
            pos_emb = self.param(
                "pos_emb",
                nn.initializers.normal(stddev=0.02),
                (seq_len, num_feats),
            )
        elif self.encoding_type == "sinusoidal":
            # Adjusted to multi-device setting.
            position = jnp.arange(0, seq_len, dtype=jnp.float32)[:, None]
            div_term = jnp.exp(
                jnp.arange(tp_index * num_feats, (tp_index + 1) * num_feats, 2)
                * (-np.log(10000.0) / (tp_size * num_feats))
            )
            pos_emb = jnp.stack(
                [jnp.sin(position * div_term), jnp.cos(position * div_term)], axis=-1
            )
            pos_emb = jnp.reshape(pos_emb, (seq_len, num_feats))
        else:
            raise ValueError(
                f"Unknown positional encoding type: {self.encoding_type}"
            )
        pos_emb = pos_emb.astype(
            x.dtype
        )  # Cast to the same dtype as the input, e.g. support bfloat16.
        pos_emb = jnp.expand_dims(pos_emb, axis=range(x.ndim - 2))
        x = x + pos_emb
        return x

class InputEmbedding(nn.Module):
    seq_len: int
    shard_axis_name: str
    data_type: jnp.dtype
    vocab_size: int
    embedding_dim: int
    encoding_type: Literal["learned","sinusoidal"] = "sinusoidal"

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(self,x: jax.Array) -> jax.Array:
        num_devices = jax.lax.psum(1,self.shard_axis_name)
        x = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_dim//num_devices,
            embedding_init=nn.initializers.normal(stddev=1.0),
            dtype=self.data_type,
            name="token_embedding",
        )(x)
        x = OldPositionalEncoding(
            embedding_dim = self.embedding_dim,    
            shard_axis_name=self.shard_axis_name,
            name="PositionalEncoding")(x)
        return x
