"""Response shape constraints; evidence support still requires separate review."""


def response_schema(source_count):
    if source_count < 1:
        raise ValueError('At least one retrieved source is required.')
    claim = {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'text': {'type': 'string', 'minLength': 1},
            'source': {'type': 'string', 'enum': [f'S{i}' for i in range(1, source_count + 1)]},
            'quote': {'type': 'string', 'minLength': 1},
        },
        'required': ['text', 'source', 'quote'],
    }
    def branch(status, minimum, maximum):
        return {
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'status': {'type': 'string', 'const': status},
                'claims': {'type': 'array', 'items': claim, 'minItems': minimum, 'maxItems': maximum},
            },
            'required': ['status', 'claims'],
        }
    return {'anyOf': [branch('answered', 1, 2), branch('insufficient_evidence', 0, 0)]}
