
try:
    from typing import Any, Dict, List, TypedDict
except Exception:
    from typing import Any, Dict, List
    from typing_extensions import TypedDict

Node = TypedDict('Node', {
    'data': Dict[str, Any], 
    'edges': List[int]
    })
