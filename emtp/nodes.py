"""Node management: compact index mapping and named-node book."""

from typing import Dict, List, Optional, Union

import numpy as np


class NodeIndexer:
    """External integer node id <-> internal compact index in [0, n).

    External 0 (GND) is special: it has no compact index and is
    represented by the sentinel ``COMPACT_GND = -1``.

    Usage
    -----
    >>> idx = NodeIndexer()
    >>> idx.register(1)    # 0
    >>> idx.register(5)    # 1
    >>> idx.register(9999) # 2
    >>> idx.n
    3
    >>> idx.to_compact(5)
    1
    >>> idx.to_external(1)
    5
    >>> idx.freeze()
    >>> idx.register(100)  # RuntimeError
    """

    COMPACT_GND: int = -1

    def __init__(self) -> None:
        self._ext_to_int: Dict[int, int] = {}
        self._int_to_ext: List[int] = []
        self._frozen: bool = False

    def register(self, ext: int) -> int:
        """Register an external node id and return its compact index.

        Returns ``COMPACT_GND`` for ground (ext == 0).
        Repeated calls with the same id are idempotent.
        Raises ``RuntimeError`` if frozen and *ext* is new.
        """
        if ext == 0:
            return self.COMPACT_GND
        if self._frozen and ext not in self._ext_to_int:
            raise RuntimeError(f"NodeIndexer is frozen; cannot add new node {ext}")
        if ext not in self._ext_to_int:
            self._ext_to_int[ext] = len(self._int_to_ext)
            self._int_to_ext.append(ext)
        return self._ext_to_int[ext]

    def to_compact(self, ext: int) -> int:
        """External id -> compact index.  Raises ``KeyError`` for unknown ids."""
        if ext == 0:
            return self.COMPACT_GND
        return self._ext_to_int[ext]

    def to_external(self, compact: int) -> int:
        """Compact index -> external id."""
        return self._int_to_ext[compact]

    def freeze(self) -> None:
        """Lock the indexer so no new external ids can be registered."""
        self._frozen = True

    @property
    def n(self) -> int:
        """Number of compact nodes (0-indexed, exclusive upper bound)."""
        return len(self._int_to_ext)

    @property
    def externals(self) -> List[int]:
        """Copy of registered external ids in compact order."""
        return list(self._int_to_ext)


class NodeBook:
    """String node-name manager that maps names to integer node ids.

    Design notes
    ------------
    - Integer nodes (0, 1, 2, ...) pass through unchanged (backward compat).
    - String node names auto-allocate the next available integer id on first use.
    - Special names (GND/0/ground etc.) are uniformly mapped to 0 (ground).
    - Supports manual binding (reserve) and aliasing (alias).

    Usage
    -----
    >>> book = NodeBook()
    >>> book.get("T1.tower_top")   # auto-allocates 1
    1
    >>> book.get("GND")            # ground node
    0
    >>> book.get(5)                # integer passthrough
    5
    """

    GROUND_NAMES = {"0", "GND", "gnd", "ground", "GROUND", "Ground"}

    def __init__(self, start: int = 1):
        if start < 1:
            raise ValueError("NodeBook start must be >= 1")
        self._next = int(start)
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}

    def get(self, node: Union[str, int, np.integer]) -> int:
        """Return integer node id.  Auto-allocates for new string names."""
        if isinstance(node, (int, np.integer)):
            if int(node) < 0:
                raise ValueError(f"Node id must be >= 0, got {node}")
            return int(node)

        name = str(node)
        if name in self.GROUND_NAMES:
            return 0

        if name not in self._name_to_id:
            node_id = self._next
            self._name_to_id[name] = node_id
            self._id_to_name[node_id] = name
            self._next += 1

        return self._name_to_id[name]

    def reserve(self, name: str, node_id: Optional[int] = None) -> int:
        """Manually bind a name.  Useful for compatibility with existing integer node models."""
        if name in self.GROUND_NAMES:
            return 0

        if node_id is None:
            return self.get(name)

        node_id = int(node_id)
        if node_id <= 0:
            raise ValueError("Non-ground node id must be > 0")

        if name in self._name_to_id and self._name_to_id[name] != node_id:
            raise ValueError(
                f"Node name {name!r} already bound to {self._name_to_id[name]}"
            )

        if node_id in self._id_to_name and self._id_to_name[node_id] != name:
            raise ValueError(
                f"Node id {node_id} already bound to {self._id_to_name[node_id]!r}"
            )

        self._name_to_id[name] = node_id
        self._id_to_name[node_id] = name
        self._next = max(self._next, node_id + 1)
        return node_id

    def alias(self, alias_name: str, existing: Union[str, int]) -> int:
        """Add an alias for an existing node (multiple names share the same id).

        ``_id_to_name`` keeps only the first / primary name; ``_name_to_id``
        adds the alias_name -> node_id mapping.
        """
        node_id = self.get(existing)
        if node_id == 0:
            return 0

        if alias_name in self.GROUND_NAMES:
            raise ValueError(f"Cannot use reserved name {alias_name!r} as alias")

        if alias_name in self._name_to_id:
            if self._name_to_id[alias_name] != node_id:
                raise ValueError(
                    f"Alias {alias_name!r} already bound to node "
                    f"{self._name_to_id[alias_name]}, cannot rebind to {node_id}"
                )
            return node_id

        self._name_to_id[alias_name] = node_id
        return node_id

    def name_of(self, node_id: int) -> Optional[str]:
        """Look up a name by integer node id."""
        node_id = int(node_id)
        if node_id == 0:
            return "GND"
        return self._id_to_name.get(node_id)

    def as_dict(self) -> Dict[str, int]:
        return dict(self._name_to_id)

    def __len__(self) -> int:
        return len(self._name_to_id)

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_id or name in self.GROUND_NAMES

    def dump(self) -> None:
        """Print the node mapping table."""
        print("Node mapping table:")
        print(f"  {'ID':>6s}  Name")
        print(f"  {'-'*6}  {'-'*30}")
        print(f"  {0:>6d}  GND")
        for name, node_id in sorted(self._name_to_id.items(),
                                    key=lambda kv: kv[1]):
            print(f"  {node_id:>6d}  {name}")
