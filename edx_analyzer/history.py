"""Bounded undo/redo stack for discrete style changes (colors, markers,
legend position, grid). Callers snapshot state themselves; this class just
stores the before/after pairs."""


class ChangeHistory:
    def __init__(self, limit=50):
        self._undo = []
        self._redo = []
        self._limit = limit

    def push(self, old_state):
        """Record the state that was current right before a change."""
        self._undo.append(old_state)
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current_state):
        """Given the current state, pop the previous one and return it (or
        None if there's nothing to undo). The current state is pushed onto
        the redo stack."""
        if not self._undo:
            return None
        self._redo.append(current_state)
        return self._undo.pop()

    def redo(self, current_state):
        if not self._redo:
            return None
        self._undo.append(current_state)
        return self._redo.pop()

    def reset(self):
        self._undo.clear()
        self._redo.clear()

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)
