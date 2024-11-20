#basic KVblock with partitioning support for concatenating and retrieving KV tensors for inference ops

import jax
import flax.linen as nn
from flax.linen import Variable
from typing import Any, Dict, Tuple, Callable, Optional, Mapping
import jax.numpy as jnp

cache_axis_name = ("kvcache_seqlen","kvcache_heads","kvcache_batch","kvcache_attndim")


class kvcache(nn.Module):
    max_sequence_len: int
    max_prefill_predict_len: int
    prefill_axis_names: Tuple[str,...]
    append_axis_names: Tuple[str,...]
    prefill_dim_order: Tuple[int,...] 
    append_dim_order:  Tuple[int,...]
    dtype: jnp.dtype

    def setup(self):
        super().setup()

    @nn.compact
    def alloc_kvcache_variables(self,mode: str, batch: int , seqlen: int , heads: int, attn_dim: int):

        """  basic cache block for inference 
        Arguments:
          mode : str  prefill or append op
          batch,heads,seqlen,attn_dim :  shape of kv tensor
          head = kv_heads 
                                     seqlen = for prefill same as max predict length
                                     seqlen = for append  output_token_len - prefill_predict_length
          dtype: tensor elements  data type in kv cache

        return variables with shard partitioning

        """ 

        seqlen = self.max_prefill_predict_len if mode == "prefill" else self.max_sequence_len - self.max_prefill_predict_len 
        kvcache_logical_shape = (batch, seqlen, heads, attn_dim)


        key_name = "kvcached_prefill_key" if mode == "prefill" else "kvcached_append_key"
        value_name = "kvcached_prefill_value" if mode == "prefill" else "kvcached_append_value"
        segment_name = "kvcached_prefill_segment_id" if mode == "prefill" else "kvcached_append_segment_id"
        shard_axis_name = self.prefill_axis_names if mode == "prefill" else self.append_axis_names
        shard_axis_order = self.prefill_dim_order if mode == "prefill" else self.append_dim_order

        kvcache_axis_names = tuple([shard_axis_name[i] for i in shard_axis_order])
        kvcache_shape = tuple([kvcache_logical_shape[i] for i in shard_axis_order])   # reorder the tensor shape in kvcache ???

    
        cached_key_var = self.variable(
            "kvcache",
            key_name,
            nn.with_logical_partitioning(jnp.zeros, kvcache_axis_names),
            kvcache_shape,
            self.dtype,
        )
        cached_value_var = self.variable(
            "kvcache",
            value_name,
            nn.with_logical_partitioning(jnp.zeros, kvcache_axis_names),
            kvcache_shape,
            self.dtype,
        )
        seqlen = self.max_prefill_predict_len if mode == "prefill" else self.max_sequence_len
        cached_segment_id_var = self.variable(
           "kvcache",
           segment_name,
           nn.with_logical_partitioning(jnp.zeros, ("kvcache_batch","kvcache_seqlen")),
           (kvcache_logical_shape[0], seqlen),
           jnp.int32,
        )

        #cached_key_scale_var = None
        #cached_value_scale_var = None

        if mode == "append":
            cached_lengths_var = self.variable(
                "kvcache",
                "kvcached_append_lengths",
                nn.with_logical_partitioning(jnp.zeros, ("kvcache_batch",)),
                (kvcache_logical_shape[0],),
                jnp.int32,
            )
            cached_index_var = self.variable(
                "kvcache",
                "kvcached_append_index",
                nn.with_logical_partitioning(jnp.zeros, ()),
                (1,),
                jnp.int32,
            )
            return cached_key_var, cached_value_var, cached_segment_id_var, cached_lengths_var,cached_index_var
        
        return cached_key_var, cached_value_var, cached_segment_id_var


    @nn.compact
    def llama_kvcache(self,mode: str, batch: int , seqlen: int , heads: int, attn_dim: int):
        """  basic cache block for inference 
        Arguments:
          mode : str  prefill or append op
          batch,heads,seqlen,attn_dim :  shape of kv tensor
          head = kv_heads 
                                     seqlen = for prefill same as max predict length
                                     seqlen = for append  output_token_len - prefill_predict_length
          dtype: tensor elements  data type in kv cache

        return variables with shard partitioning

        """ 

        seqlen = self.max_sequence_len
        kvcache_logical_shape = (batch, seqlen, heads, attn_dim)


        key_name = "kvcached_prefill_key" if mode == "prefill" else "kvcached_append_key"
        value_name = "kvcached_prefill_value" if mode == "prefill" else "kvcached_append_value"
        shard_axis_name = self.prefill_axis_names if mode == "prefill" else self.append_axis_names
        shard_axis_order = self.prefill_dim_order if mode == "prefill" else self.append_dim_order

        kvcache_axis_names = tuple([shard_axis_name[i] for i in shard_axis_order])
        kvcache_shape = tuple([kvcache_logical_shape[i] for i in shard_axis_order])   # reorder the tensor shape in kvcache ???
        self.cached_key = self.variable(
            "kvcache",
            key_name,
            nn.with_logical_partitioning(jnp.zeros, kvcache_axis_names),
            kvcache_shape,
            self.dtype,
        )
        self.cached_value = self.variable(
            "kvcache",
            value_name,
            nn.with_logical_partitioning(jnp.zeros, kvcache_axis_names),
            kvcache_shape,
            self.dtype,
        )


    def llama_prefill(self,key: jax.Array, value: jax.Array, start_pos: int):

        batch,seqlen = key.shape
        assert(key.shape == value.shape, "key and value shape must match")
        assert(key.dtype == value.dtype, "Key and value dtpes should match")

        self.alloc_kvcache("prefill",batch,seqlen,heads,attn_dim)

        _key = jnp.transpose(key,self.prefill_dim_order)
        _value = jnp.transpose(value,self.prefill_dim_order)

        self.cached_key.value[:batch, start_pos : start_pos + seqlen] = _key
        self.cached_value.value[:batch, start_pos : start_pos + seqlen] = _value 

    def llama_append(self,key: jax.Array, value: jax.Array, start_pos: int):
        batch,seqlen = key.shape
        assert(key.shape == value.shape, "key and value shape must match")
        assert(key.dtype == value.dtype, "Key and value dtpes should match")

        self.alloc_kvcache("append",batch,seqlen,heads,attn_dim)
        _key = jnp.transpose(key,self.prefill_dim_order)
        _value = jnp.transpose(value,self.prefill_dim_order)

        ##write 
        self.cached_key.value[:batch, start_pos : start_pos + seqlen] = _key
        self.cached_value.value[:batch, start_pos : start_pos + seqlen] = _value 

        #read

        _key = self.cached_key.value[:batch : start_pos+seqlen]
        _value = self.cached_value.value[:batch: start_pos+seqlen]

        #FIXME where do we handle kv_heads< q_heads
        ## in attention layer
        return _key, _value


    def kvcache_prefill(self, key: jax.Array, value: jax.Array, decoder_segment_ids: jax.Array):

        """ prefill mode, update kvcache with inputs 
        Args:
        key : jax.Array key tensor
        value : jax.Array value tensor
        segment_ids :  jax.array segment ids 
        returns:
        tuple[key,value,segment_ids]
        """

        batch,seqlen,heads,attn_dim = key.shape
        assert(key.shape == value.shape, "key and value shape must match")
        assert(key.dtype == value.dtype, "Key and value dtpes should match")

        _prefill_key_var, _prefill_value_var, _prefill_segment_ids_var = self.alloc_kvcache_variables("prefill",batch,seqlen,heads,attn_dim)

        kvcache_key = jnp.transpose(key,self.prefill_dim_order)
        kvcache_value = jnp.transpose(value,self.prefill_dim_order)

        _prefill_key_var.value = kvcache_key
        _prefill_value_var.value = kvcache_value
        if decoder_segment_ids is not None:
            _prefill_segment_ids_var = decoder_segment_ids

        return key, value, decoder_segment_ids

    def append_key_value(self,key_token: jax.Array, value_token: jax.Array,kvcached_key_var: nn.Variable, kvcached_value_var: nn.Variable, token_indices: jax.Array, lengths: jax.Array):
        """Adds a single token's results to the ar kv cache
        Args:
            key_token  (jax.Array): Key token(#1) to add to the cache
            value_token (jax.Array): Value token(#1) to add to the cache
            cached_key_ (nn.Variable: Cached keys to add new token key to
            cached_value (nn.Variable: Cached values to add new token value to
            token_indices (Array): Location of the new token within the cache

        Returns:
                tuple[Array, Array]: Updated caches for key and value with new token info added
        """

        # In order to update the key, value caches with the current key and
        # value, we reshape the key_token and value_token
        kvcache_key = jnp.transpose(key_token, self.append_dim_order)
        kvcache_value = jnp.transpose(value_token, self.append_dim_order)
        ## reorder axis names based on dimension order (dimension order is compute order kvcache kept in differnet dimension order
        kvcache_axis_names = tuple([self.append_axis_names[i] for i in self.append_dim_order])
        kvcache_update_idx = jnp.squeeze(token_indices)
        token_indices = token_indices.astype(int)
        ##append key,value token to kvcache
        kvcached_key_var.value = jax.lax.dynamic_update_index_in_dim(
        kvcached_key_var.value, kvcache_key, kvcache_update_idx, kvcache_axis_names.index("kvcache_seqlen") 
        )
        kvcached_value_var.value = jax.lax.dynamic_update_index_in_dim(
           kvcached_value_var.value, kvcache_value, kvcache_update_idx, kvcache_axis_names.index("kvcache_seqlen") 
        )

        kvcached_key_var.value = nn.with_logical_constraint(kvcached_key_var.value, kvcache_axis_names)
        kvcached_value_var.value = nn.with_logical_constraint(kvcached_value_var.value, kvcache_axis_names)

        return

    def get_cached_values(self, cache_var, kvcache_axis_order) -> jax.Array:
        cache_value = jnp.asarray(cache_var.value)
        cache_value_in_logical_shape = jax.tree.map(lambda x: jax.numpy.moveaxis(x, (0, 1, 2, 3), kvcache_axis_order), cache_value)
        return cache_value_in_logical_shape

    def kvcache_append(
        self,
        key: jax.Array,
        value: jax.Array,
        decoder_segment_ids: jax.Array
        ):
        """In append mode, update the cache and return the full cache
        Args:
            key: in shape [b, 1, n, d].
            value: in shape [b, 1, n, d].
            decoder_segment_ids: [b, 1] -- marking segment ids for tokens

        Returns:
            tuple of (key, value, segment_id) for both prefill and ar cache,
        """

        batch, sequence, heads, attn_dim = key.shape
        assert(sequence == 1, f"Sequence length should be 1 during autoregression, got {sequence=}")
        is_initialized = self.has_variable("kvcache", "kvcache_append_index")
        assert(is_initialized, "Error, we can't do autoregression if we haven't seeded the KV Cache.")

        _append_key_var, _append_value_var, _append_segment_id_var, _append_index_var, _append_lengths_var = (
             self.alloc_kvcache_variables("append",batch, sequence, heads, attn_dim)
        )
        self.append_key_value(key,value,_append_key_var,_append_value_var,_append_index_var.value, _append_lengths_var.value)

        active_indicator = jnp.zeros((batch, 1), dtype=jnp.int32) + 1
        _append_segment_id_var.value = jax.lax.dynamic_update_index_in_dim(
            _append_segment_id_var.value, active_indicator, jnp.squeeze(_append_index_var.value), 1
        )
        _append_index_var.value = jnp.mod(
             _append_index_var.value + 1, self.max_sequence_len - self.max_prefill_predict_length
        )
        _append_lengths_var.value = _append_lengths_var.value.at[:].add(1)

        # The below retrieves the existing prefill cache variables, not creating new ones
        _prefill_key_var, _prefill_value_var, _prefill_segment_id_var = self.alloc_kvcache_variables(
             batch, heads, sequence, kv_head_size
        )

        cached_prefill = (
            self.get_cached_values(_prefill_key_vars, self.prefill_dim_order),
            self.get_cached_values(cached_prefill_value_vars, self.prefill_dim_order),
            _prefill_segment_id_var.value,
        )

        cached_append = (
            self.get_cached_values(_append_key_var, self.append_dim_order),
            self.get_cached_values(_append_value_var, self.append_dim_order),
            _append_segment_id_var.value,
            _append_lengths_var.value,
        )
        return cached_prefill, cached_append
    @nn.compact
    def __call__(self, key: jax.Array, value: jax.Array, decoder_segment_ids: jax.Array, mode: str) -> tuple:

        """ KV cache appends and returns 
            Arguments: 
               key : shape [batch, seqlen, heads,attn_dim]   heads : kv_div
               value : shape [batch, seqlen, heads,attn_dim]   heads : kv_div
               mode : model mode

            returns:
                tuple of KV tensor
        """
        assert(key.shape == value.shape)


        if mode == "prefill":
            #prefill mode, fillup cache with zeros and return prefill"
            return self.kvcache_prefill(key,value,decoder_segment_ids), None
        elif mode == "append":
            return self.kvcache_append(key,value,decoder_segment_ids)
        elif mode == "training":
            return (key,value, decoder_segment_ids),None
        else:
            raise ValueError(f"Invalue Model mode passed ... {mode}")

class llama_kvcache(nn.Module):
    max_sequence_len: int
    prefill_axis_names: Tuple[str,...]
    append_axis_names: Tuple[str,...]
    prefill_dim_order: Tuple[int,...] 
    append_dim_order:  Tuple[int,...]
    dtype: jnp.dtype
    num_heads : int
    batch_size: int
    attn_dim: int
    mode: str = "prefill"

    def setup(self):
        super().setup()
        if (self.mode != "Training"):
            self.alloc_kvcache("prefill",self.batch_size,self.max_sequence_len,self.num_heads,self.attn_dim)
            self.alloc_kvcache("append",self.batch_size,self.max_sequence_len,self.num_heads,self.attn_dim)

    #@nn.compact
    def alloc_kvcache(self,mode: str, batch: int , seqlen: int , heads: int, attn_dim: int):
        """  basic cache block for inference 
        Arguments:
          mode : str  prefill or append op
          batch,heads,seqlen,attn_dim :  shape of kv tensor
          head = kv_heads 
                                     seqlen = for prefill same as max predict length
                                     seqlen = for append  output_token_len - prefill_predict_length
          dtype: tensor elements  data type in kv cache

        return variables with shard partitioning
        """

        seqlen = self.max_sequence_len
        kvcache_logical_shape = (batch, seqlen, heads, attn_dim)


        key_name = "kvcached_prefill_key" if mode == "prefill" else "kvcached_append_key"
        value_name = "kvcached_prefill_value" if mode == "prefill" else "kvcached_append_value"
        shard_axis_name = self.prefill_axis_names if mode == "prefill" else self.append_axis_names
        shard_axis_order = self.prefill_dim_order if mode == "prefill" else self.append_dim_order

        kvcache_axis_names = tuple([shard_axis_name[i] for i in shard_axis_order]) if shard_axis_name is not None else None
        kvcache_shape = tuple([kvcache_logical_shape[i] for i in shard_axis_order])   # reorder the tensor shape in kvcache ???
        #key = self.make_rng('kvcache')
        self.cached_key = self.variable(
            "kvcache",
            key_name,
            nn.with_logical_partitioning(jnp.zeros, kvcache_axis_names),
            kvcache_shape,
            self.dtype,
        )
        self.cached_value = self.variable(
            "kvcache",
            value_name,
            nn.with_logical_partitioning(jnp.zeros, kvcache_axis_names),
            kvcache_shape,
            self.dtype,
        )


    def llama_prefill(self,key: jax.Array, value: jax.Array, start_pos: int):

        batch,seqlen,heads,attn_dim = key.shape
        #print(key.shape)
        assert(key.shape == value.shape, "key and value shape must match")
        assert(key.dtype == value.dtype, "Key and value dtpes should match")

        #self.alloc_kvcache("prefill",batch,seqlen,heads,attn_dim)

        _key = jnp.transpose(key,self.prefill_dim_order)
        _value = jnp.transpose(value,self.prefill_dim_order)
        #print(_key.shape)

        #print(self.cached_key.value.shape)
        #self.cached_key.value[:batch, start_pos : start_pos + seqlen] = _key
        #self.cached_value.value[:batch, start_pos : start_pos + seqlen] = _value 
        self.cached_key.value = self.cached_key.value.at[:batch, start_pos : start_pos + seqlen].set(_key)
        self.cached_value.value = self.cached_key.value.at[:batch, start_pos : start_pos + seqlen].set(_value)

        return key,value,start_pos

    def llama_append(self,key: jax.Array, value: jax.Array, start_pos: int):
        batch,seqlen,heads,attn_dim = key.shape
        assert(key.shape == value.shape, "key and value shape must match")
        assert(key.dtype == value.dtype, "Key and value dtpes should match")

        #self.alloc_kvcache("append",batch,seqlen,heads,attn_dim)
        _key = jnp.transpose(key,self.prefill_dim_order)
        _value = jnp.transpose(value,self.prefill_dim_order)

        ##write 
        self.cached_key.value[:batch, start_pos : start_pos + seqlen] = _key
        self.cached_value.value[:batch, start_pos : start_pos + seqlen] = _value 

        #read

        _key = self.cached_key.value[:batch : start_pos+seqlen]
        _value = self.cached_value.value[:batch: start_pos+seqlen]

        #FIXME where do we handle kv_heads< q_heads
        ## in attention layer
        return _key, _value

    @nn.compact
    def __call__(self, key: jax.Array, value: jax.Array, start_pos: int, mode: str) -> tuple:

        """ KV cache appends and returns 
            Arguments: 
               key : shape [batch, seqlen, heads,attn_dim]   heads : kv_div
               value : shape [batch, seqlen, heads,attn_dim]   heads : kv_div
               mode : model mode

            returns:
                tuple of KV tensor
        """
        assert(key.shape == value.shape)


        if mode == "prefill":
            #prefill mode, fillup cache with zeros and return prefill"
            return self.llama_prefill(key,value,start_pos)
        elif mode == "append":
            return self.llama_append(key,value,start_pos)
        elif mode == "training":
            return (key,value,start_pos)
        else:
            raise ValueError(f"Invalue Model mode passed ... {mode}")
