"""MHA/GQA adapter using a pool's existing physical copy hooks."""


class MHAStageAdapter:
    def __init__(self, host_pool):
        self.host_pool = host_pool

    def bytes_for_chunk(self, chunk):
        return chunk.byte_count

    def enqueue_d2h(self, chunk, stream):
        return self.host_pool.backup_stage_chunk(chunk, stream)

    def enqueue_h2d_layer(self, chunk, layer_id, stream):
        return self.host_pool.load_stage_chunk(chunk, layer_id, stream)
