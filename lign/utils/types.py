
try:
    from typing import TypeVar, Dict, List, TypedDict
except Exception:
    from typing import TypeVar, Dict, List
    from typing_extensions import TypedDict

T = TypeVar('T')

Node = TypedDict('Node', {
    'data': Dict[str, T], 
    'edges': List[int]
    })
