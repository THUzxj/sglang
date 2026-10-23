"""Composite adapter with all child pools participating in one ACK."""


class HybridStageAdapter:
    def __init__(self, adapters):
        self.adapters = tuple(adapters)

    def bytes_for_chunk(self, chunk):
        return chunk.byte_count

    def enqueue_d2h(self, chunk, stream):
        events = [adapter.enqueue_d2h(chunk, stream) for adapter in self.adapters]
        return _combine_events(events)

    def enqueue_h2d_layer(self, chunk, layer_id, stream):
        events = [adapter.enqueue_h2d_layer(chunk, layer_id, stream) for adapter in self.adapters]
        return _combine_events(events)


def _combine_events(events):
    events = [event for event in events if event is not None]
    if not events:
        return None
    if len(events) == 1:
        return events[0]

    class AggregateEvent:
        def query(self):
            return all(not hasattr(event, "query") or event.query() for event in events)

        def synchronize(self):
            for event in events:
                if hasattr(event, "synchronize"):
                    event.synchronize()

    return AggregateEvent()
